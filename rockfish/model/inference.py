import torch
from torch.nn import DataParallel
import torch.profiler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import argparse

from datasets import read_offsets, parse_ctgs, read_example, MappingEncodings
from model import Rockfish

ENCODING = {b: i for i, b in enumerate('ACGT')}


def parse_gpus(string):
    if string is None:
        return None

    gpus = string.strip().split(',')
    return [int(g) for g in gpus]


class RFDataset(Dataset):
    def __init__(self, path: str, features: int, seq_len: int,
                 block_size: int) -> None:
        super(Dataset, self).__init__()

        self.seq_len = seq_len
        self.block_size = block_size

        self.path = path
        self.fd = None
        self.ctgs = None

        self.offsets = read_offsets(f'{path}.idx')

        self.mapping_encoding = MappingEncodings(features, self.seq_len,
                                                 block_size)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        example = read_example(self.fd, self.offsets[idx], self.seq_len)

        signal = torch.tensor(example.signal).unfold(
            -1, self.block_size, self.block_size)  # Converting to blocks

        bases = torch.tensor([ENCODING[b] for b in example.bases])
        q_indices = torch.tensor(example.q_indices)
        lengths = torch.tensor(example.lengths)

        r_pos_enc, q_pos_enc = self.mapping_encoding(lengths, q_indices)

        return example.read_id, self.ctgs[
            example.ctg], example.pos, signal, bases, r_pos_enc, q_pos_enc


def collate_fn(batch):
    read_ids, ctgs, poss, signals, bases, q_indices, lengths = zip(*batch)
    signals = pad_sequence(signals, batch_first=True)  # BxMAX_LEN
    q_indices = pad_sequence(q_indices, batch_first=True)  # [B,MAX_LEN//5]

    return read_ids, ctgs, poss, signals, torch.stack(
        bases, 0), q_indices, torch.stack(lengths, 0)


def collate_fn(batch):
    read_ids, ctgs, poss, signals, bases, r_pos_enc, q_pos_enc = zip(*batch)

    num_blocks = torch.tensor([len(s) for s in signals])  # B
    signals = pad_sequence(signals,
                           batch_first=True)  # [B, MAX_LEN, BLOCK_SIZE]
    r_pos_enc = pad_sequence(r_pos_enc, batch_first=True)  # [B, MAX_LEN, E//4]
    q_pos_enc = pad_sequence(q_pos_enc, batch_first=True)  # [B, MAX_LEN, E//4]

    return read_ids, ctgs, poss, signals, torch.stack(
        bases, 0), r_pos_enc, q_pos_enc, num_blocks


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.fd = open(dataset.path, 'rb')
    dataset.ctgs = parse_ctgs(dataset.fd)


@torch.no_grad()
def inference(args):
    model = Rockfish.load_from_checkpoint(args.ckpt_path).eval()

    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    if gpus is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpus[0]}')
        if len(gpus) > 1:
            model = DataParallel(model, device_ids=gpus)
    else:
        device = torch.device('cpu')
    model.to(device)

    data = RFDataset(args.data_path, 384, 31, 5)
    if args.workers == 0:
        data.fd = open(data.path, 'rb')
        data.ctgs = parse_ctgs(data.fd)

    loader = DataLoader(data,
                        args.batch,
                        False,
                        num_workers=args.workers,
                        collate_fn=collate_fn,
                        worker_init_fn=worker_init_fn,
                        pin_memory=True)

    with open(
            args.output, 'w'
    ) as f, tqdm(total=len(data)) as pbar, torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=3,
                                             warmup=3,
                                             active=10,
                                             repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                '/raid-ssd/stanojevicd/dna-mod/NA12878/no_tombo/log/rockfish'),
            record_shapes=True,
            with_stack=True) as prof:
        for i, (ids, ctgs, poss, signals, bases, r_pos_enc, q_pos_enc,
                num_blocks) in enumerate(loader):
            if i >= (3 + 3 + 10) * 2:
                break

            signals, bases, r_pos_enc, q_pos_enc, num_blocks = (
                signals.to(device), bases.to(device), r_pos_enc.to(device),
                q_pos_enc.to(device), num_blocks.to(device))

            logits = model(signals, bases, r_pos_enc, q_pos_enc,
                           num_blocks).cpu().numpy()  # N

            for id, ctg, pos, logit in zip(ids, ctgs, poss, logits):
                print(id, ctg, pos, logit, file=f, sep='\t')
            pbar.update(len(logits))
            prof.step()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('-o', '--output', type=str, default='preditions.tsv')
    parser.add_argument('-d', '--gpus', default=None)
    parser.add_argument('-t', '--workers', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    inference(args)
