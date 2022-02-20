import torch
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import math
import struct
import argparse

from datasets import read_offsets, parse_ctgs, Example, MappingEncodings
from model import Rockfish

ENCODING = {b: i for i, b in enumerate('ACGT')}


def parse_gpus(string):
    if string is None:
        return None

    gpus = string.strip().split(',')
    return [int(g) for g in gpus]


class RFDataset(IterableDataset):
    def __init__(self,
                 path: str,
                 batch_size: int,
                 window: int = 15,
                 block_size: int = 5) -> None:
        super(IterableDataset, self).__init__()

        self.batch_size = batch_size
        self.seq_len = (2 * window) + 1
        self.block_size = block_size

        self.mapping_encodings = MappingEncodings(self.seq_len,
                                                  self.block_size)

        self.path = path
        self.fd = None
        self.ctgs = None

        self.offsets = read_offsets(f'{path}.idx')
        self.start = 0
        self.end = len(self.offsets)

    def __iter__(self):
        bins = [list() for _ in range(9)]
        stored = 0

        self.fd.seek(self.offsets[self.start])
        for _ in range(self.start, self.end):
            example = self.read_iter_example()
            bin = len(example.q_indices) // 10 - 3
            bins[bin].append(example)
            stored += 1

            if len(bins[bin]) >= self.batch_size:  # bin is full, emit examples
                for example in bins[bin]:
                    signal = torch.tensor(example.signal,
                                          dtype=torch.half).unfold(
                                              -1, self.block_size,
                                              self.block_size)
                    bases = torch.tensor([ENCODING[b] for b in example.bases])
                    q_indices = torch.tensor(example.q_indices)
                    lengths = torch.tensor(example.lengths)

                    r_pos_enc = self.mapping_encodings(lengths)

                    yield example.read_id, self.ctgs[
                        example.
                        ctg], example.pos, signal, bases, r_pos_enc, q_indices

                stored -= len(bins[bin])
                bins[bin].clear()
            elif stored >= 4 * self.batch_size:
                for bin in bins:
                    for example in bin:
                        signal = torch.tensor(example.signal,
                                              dtype=torch.half).unfold(
                                                  -1, self.block_size,
                                                  self.block_size)
                        bases = torch.tensor(
                            [ENCODING[b] for b in example.bases])
                        q_indices = torch.tensor(example.q_indices)
                        lengths = torch.tensor(example.lengths)

                        r_pos_enc = self.mapping_encodings(lengths)

                        yield example.read_id, self.ctgs[
                            example.
                            ctg], example.pos, signal, bases, r_pos_enc, q_indices

                    bin.clear()
                stored = 0

    def read_iter_example(self):
        read_id, ctg, pos, n_points, q_indices_len = struct.unpack(
            '=36sHIHH', self.fd.read(46))

        n_bytes = 2 * n_points + 2 * q_indices_len + 3 * self.seq_len
        data = struct.unpack(
            f'={n_points}e{q_indices_len}H{self.seq_len}H{self.seq_len}s',
            self.fd.read(n_bytes))
        event_len_start = n_points + q_indices_len

        return Example(read_id.decode(), ctg, pos, data[:n_points],
                       data[n_points:event_len_start],
                       data[event_len_start:-1], data[-1].decode())


def collate_fn(batch):
    read_ids, ctgs, poss, signals, bases, r_pos_enc, q_indices = zip(*batch)

    num_blocks = torch.tensor([len(s) for s in signals])
    signals = pad_sequence(signals, batch_first=True)  # BxMAX_LEN
    r_pos_enc = pad_sequence(r_pos_enc, batch_first=True)
    q_indices = pad_sequence(q_indices, batch_first=True)  # [B,MAX_LEN//5]

    return read_ids, ctgs, poss, signals, torch.stack(
        bases, 0), r_pos_enc, q_indices, num_blocks


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.fd = open(dataset.path, 'rb')
    dataset.ctgs = parse_ctgs(dataset.fd)

    per_worker = int(
        math.ceil(len(dataset.offsets) / float(worker_info.num_workers)))
    dataset.start = worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, len(dataset.offsets))


@torch.no_grad()
def inference(args):
    model = Rockfish.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.freeze()

    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    if gpus is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpus[0]}')
        if len(gpus) > 1:
            model = DataParallel(model, device_ids=gpus)
    else:
        device = torch.device('cpu')
    print(device)
    model.to(device)

    data = RFDataset(args.data_path, args.batch_size)
    loader = DataLoader(data,
                        args.batch_size,
                        False,
                        num_workers=args.workers,
                        collate_fn=collate_fn,
                        worker_init_fn=worker_init_fn,
                        pin_memory=True)

    with open(args.output, 'w') as f, tqdm(
            total=len(data.offsets)) as pbar, torch.cuda.amp.autocast():
        for ids, ctgs, poss, signals, bases, r_pos_enc, q_indices, num_blocks in loader:
            signals, bases, r_pos_enc, q_indices, num_blocks = (
                signals.to(device), bases.to(device), r_pos_enc.to(device),
                q_indices.to(device), num_blocks.to(device))

            logits = model(signals, bases, r_pos_enc, q_indices,
                           num_blocks).cpu().numpy()  # N

            for id, ctg, pos, logit in zip(ids, ctgs, poss, logits):
                print(id, ctg, pos, logit, file=f, sep='\t')
            pbar.update(len(logits))


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('-o', '--output', type=str, default='preditions.tsv')
    parser.add_argument('-d', '--gpus', default=None)
    parser.add_argument('-t', '--workers', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    inference(args)
