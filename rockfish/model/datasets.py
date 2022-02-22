import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from dataclasses import dataclass
from io import BufferedReader
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
    for _ in range(n_ctgs):
        ctg_name_len = int.from_bytes(fd.read(1), byteorder=sys.byteorder)
        ctg_name = struct.unpack(f'={ctg_name_len}s', fd.read(ctg_name_len))[0]
        ctgs.append(ctg_name.decode())

    return ctgs


@dataclass
class Example:
    read_id: str
    ctg: int
    pos: int
    signal: np.ndarray
    q_indices: List[int]
    lengths: List[int]
    bases: str


def read_example(fd: BufferedReader, offset: int, seq_len: int) -> Example:
    fd.seek(offset)

    read_id, ctg, pos, n_points, q_indices_len = struct.unpack(
        '=36sHIHH', fd.read(46))

    n_bytes = 2 * n_points + 2 * q_indices_len + 3 * seq_len
    data = struct.unpack(f'={n_points}e{q_indices_len}H{seq_len}H{seq_len}s',
                         fd.read(n_bytes))
    event_len_start = n_points + q_indices_len

    return Example(read_id.decode(), ctg, pos, data[:n_points],
                   data[n_points:event_len_start], data[event_len_start:-1],
                   data[-1].decode())


class MappingEncodings:
    def __init__(self, ref_len: int, block_size: int) -> None:
        self.block_size = block_size
        self.r_pos = torch.arange(0, ref_len)  # T

    def __call__(self, e_lens: torch.Tensor) -> torch.Tensor:
        block_lengths = torch.div(e_lens,
                                  self.block_size,
                                  rounding_mode='floor')
        r_idx = torch.repeat_interleave(self.r_pos, block_lengths)  # S_B

        return r_idx


class RFDataset(Dataset):
    def __init__(self, path: str, labels: str, seq_len: int,
                 block_size: int) -> None:
        super(Dataset, self).__init__()

        self.seq_len = seq_len
        self.block_size = block_size

        self.path = path
        self.fd = None
        self.ctgs = None

        self.offsets = read_offsets(f'{path}.idx')
        self.labels = Labels(labels)

        self.mapping_encoding = MappingEncodings(self.seq_len, block_size)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        example = read_example(self.fd, self.offsets[idx], self.seq_len)

        signal = torch.tensor(example.signal).unfold(
            -1, self.block_size, self.block_size)  # Converting to blocks

        bases = torch.tensor([ENCODING[b] for b in example.bases])
        q_indices = torch.tensor(example.q_indices)
        lengths = torch.tensor(example.lengths)
        label = self.labels.get_label(example.read_id, self.ctgs[example.ctg],
                                      example.pos)

        r_pos_enc = self.mapping_encoding(lengths)

        return signal, bases, r_pos_enc, q_indices, label


def collate_fn(batch):
    signals, bases, r_pos_enc, q_pos_enc, labels = zip(*batch)

    num_blocks = torch.tensor([len(s) for s in signals])  # B
    signals = pad_sequence(signals,
                           batch_first=True)  # [B, MAX_LEN, BLOCK_SIZE]
    r_pos_enc = pad_sequence(r_pos_enc, batch_first=True)  # [B, MAX_LEN]
    q_pos_enc = pad_sequence(q_pos_enc, batch_first=True)  # [B, MAX_LEN]

    return signals, torch.stack(
        bases, 0), r_pos_enc, q_pos_enc, num_blocks, torch.tensor(labels)


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.fd = open(dataset.path, 'rb')
    dataset.ctgs = parse_ctgs(dataset.fd)


class RFDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: str,
                 train_labels: str,
                 val_data: str,
                 val_labels: str,
                 train_batch_size: int = 256,
                 val_batch_size: int = 512,
                 seq_len: int = 31,
                 block_size: int = 5) -> None:
        super(RFDataModule, self).__init__()

        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.seq_len = seq_len
        self.block_size = block_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_ds = RFDataset(self.train_data, self.train_labels,
                                      self.seq_len, self.block_size)
            self.val_ds = RFDataset(self.val_data, self.val_labels,
                                    self.seq_len, self.block_size)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          self.train_batch_size,
                          True,
                          collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          self.val_batch_size,
                          collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=4,
                          pin_memory=True)
