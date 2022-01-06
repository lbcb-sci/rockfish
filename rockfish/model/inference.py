import torch
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import argparse

from datasets import read_offsets, parse_ctgs, read_example
from model import Rockfish

ENCODING = {b: i for i, b in enumerate('ACGT')}


def parse_gpus(string):
    if string is None:
        return None

    gpus = string.strip().split(',')
    return [int(g) for g in gpus]


class RFDataset(Dataset):
    def __init__(self, path=str, window: int = 12) -> None:
        super(Dataset, self).__init__()

        self.seq_len = (2 * window) + 1

        self.path = path
        self.fd = None
        self.ctgs = None

        self.offsets = read_offsets(f'{path}.idx')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        example = read_example(self.fd, self.offsets[idx], self.seq_len)

        signal = torch.tensor(example.signal)
        bases = torch.tensor([ENCODING[b] for b in example.bases])
        lengths = torch.tensor(example.lengths)

        return example.read_id, self.ctgs[
            example.ctg], example.pos, signal, bases, lengths


def collate_fn(batch):
    read_ids, ctgs, poss, signals, bases, lengths = zip(*batch)
    signals = pad_sequence(signals, batch_first=True)  # BxMAX_LEN

    return read_ids, ctgs, poss, signals, torch.stack(bases, 0), torch.stack(
        lengths, 0)


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.fd = open(dataset.path, 'rb')
    dataset.ctgs = parse_ctgs(dataset.fd)


def inference(args):
    model = Rockfish.load_from_checkpoint(args.ckpt_path).half()
    model.eval()
    model.freeze()

    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    if gpus is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpus[0]}')
        if len(gpus) > 1:
            model = DataParallel(model, device_ids=gpus)
    else:
        device = torch.device('cpu')
    model.to(device)

    data = RFDataset(args.data_path)
    loader = DataLoader(data,
                        args.batch,
                        False,
                        num_workers=args.workers,
                        collate_fn=collate_fn,
                        worker_init_fn=worker_init_fn,
                        pin_memory=True)

    with open(args.output, 'w') as f, tqdm(total=len(data)) as pbar:
        for ids, ctgs, poss, signals, bases, lens in loader:
            signals, bases, lens = (signals.to(device), bases.to(device),
                                    lens.to(device))

            logits = model(signals, lens, bases).cpu().numpy()  # N

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
    parser.add_argument('-b', '--batch', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    inference(args)