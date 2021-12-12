from dataclasses import dataclass
from io import BufferedReader
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import pytorch_lightning as pl

import struct
import sys

from typing import *

ENCODING = {b: i for i, b in enumerate('ACGT')}


class Labels:
    def __init__(self, path: str) -> None:
        self.label_for_read = {}
        self.label_for_pos = {}
        self.label_for_read_pos = {}

        with open(path, 'r') as f:
            for i, line in enumerate(f, start=1):
                data = line.strip().split('\t')

                if len(data) == 3:
                    self.label_for_read[data[0]] = int(data[2])
                elif len(data) == 4:
                    key = data[0], data[1], int(data[2])
                    if key[0] == '*':
                        self.label_for_pos[(key[1], key[2])] = int(data[3])
                    else:
                        self.label_for_read_pos[key] = int(data[3])
                else:
                    raise ValueError(f'Wrong label line {i}.')

    def get_label(self, read_id, ctg, pos):
        if read_id in self.label_for_read:
            return self.label_for_read[read_id]
        elif (ctg, pos) in self.label_for_pos:
            return self.label_for_pos[(ctg, pos)]

        return self.label_for_pos[(read_id, ctg, pos)]


def read_offsets(path: str) -> List[int]:
    with open(path, 'rb') as f:
        n_examples = int.from_bytes(f.read(4), byteorder=sys.byteorder)
        indices = [
            int.from_bytes(f.read(8), byteorder=sys.byteorder)
            for _ in range(n_examples)
        ]

        return indices


def parse_ctgs(fd: BufferedReader) -> List[str]:
    fd.seek(0)

    n_ctgs = int.from_bytes(fd.read(2), byteorder=sys.byteorder)
    ctgs = []
    for _ in n_ctgs:
        ctg_name_len = int.from_bytes(fd.read(1), byteorder=sys.byteorder)
        ctg_name = struct.unpack(f'={ctg_name_len}s', fd.read(ctg_name_len))[0]
        ctgs.append(ctg_name.decode())

    return ctgs


@dataclass
class Example:
    read_id: str
    ctg: int
    pos: int
    points: List[float]
    lengths: List[int]
    bases: str


def read_example(fd: BufferedReader, offset: int, seq_len: int) -> Example:
    fd.seek(offset)

    read_id, ctg, pos, n_points = struct.unpack('=36sHIH', fd.read(44))

    n_bytes = 2 * n_points + 3 * seq_len
    data = struct.unpack(f'={n_points}e{seq_len}H{seq_len}s', fd.read(n_bytes))

    return Example(read_id.decode(), ctg, pos, data[:n_points],
                   data[n_points:-1], data[-1].decode())


class RFDataset(Dataset):
    def __init__(self, path=str, labels=str, window: int = 12) -> None:
        super(Dataset, self).__init__()

        self.seq_len = (2 * window) + 1

        self.fd = open(path, 'rb')
        self.ctgs = parse_ctgs(self.fd)

        self.offsets = read_offsets(f'{path}.idx')
        self.labels = Labels(labels)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        example = read_example(self.fd, self.offsets[idx])

        signal = torch.tensor(example.signal)
        bases = torch.tensor([ENCODING[b] for b in example['kmer'].decode()])
        lengths = torch.tensor(example.lengths)
        label = self.labels.get_label(example.read_id, self.ctgs[example.ctg],
                                      example.pos)

        return signal, bases, lengths, label


def collate_fn(batch):
    signals, bases, lenghts, labels = zip(*batch)
    signals = pad_sequence(signals, batch_first=True)  # BxMAX_LEN
    return signals, torch.tensor(bases), torch.tensor(lenghts), torch.tensor(
        labels)


class RFDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: str,
                 train_labels: str,
                 val_data: str,
                 val_labels: str,
                 train_batch_size: int = 256,
                 val_batch_size: int = 512) -> None:
        super(RFDataModule, self).__init__()

        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_ds = RFDataset(self.train_data, self.train_labels)
            self.val_ds = RFDataset(self.val_data, self.val_labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          self.train_batch_size,
                          True,
                          collate_fn=collate_fn,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          self.val_batch_size,
                          collate_fn=collate_fn,
                          num_workers=4,
                          pin_memory=True)
