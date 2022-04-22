import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, DataLoader

import pytorch_lightning as pl

from dataclasses import dataclass
from io import BufferedReader
import struct
import sys
import math

from typing import *

ENCODING = {b: i for i, b in enumerate('ACGTN')}


class Labels:
    def __init__(self, path: str) -> None:
        self.label_for_read = {}
        self.label_for_pos = {}
        self.label_for_read_pos = {}

        with open(path, 'r') as f:
            for i, line in enumerate(f, start=1):
                data = line.strip().split('\t')

                if len(data) == 3:
                    self.label_for_read[data[0]] = float(data[2])
                elif len(data) == 4:
                    key = data[0], data[1], int(data[2])
                    if key[0] == '*':
                        self.label_for_pos[(key[1], key[2])] = float(data[3])
                    else:
                        self.label_for_read_pos[key] = float(data[3])
                else:
                    raise ValueError(f'Wrong label line {i}.')

    def get_label(self, read_id, ctg, pos):
        if read_id in self.label_for_read:
            return self.label_for_read[read_id]
        elif (ctg, pos) in self.label_for_pos:
            return self.label_for_pos[(ctg, pos)]

        return self.label_for_read_pos[(read_id, ctg, pos)]


def get_n_examples(path: str) -> int:
    with open(path, 'rb') as f:
        n_examples = int.from_bytes(f.read(4), byteorder=sys.byteorder)
        return n_examples


def read_offsets(path: str) -> List[int]:
    with open(path, 'rb') as f:
        n_examples = int.from_bytes(f.read(4), byteorder=sys.byteorder)
        indices = [
            int.from_bytes(f.read(8), byteorder=sys.byteorder)
            for _ in range(n_examples)
        ]

        return indices


def read_offsets2(path: str) -> List[int]:
    with open(path, 'rb') as f:
        n_examples = int.from_bytes(f.read(4), byteorder=sys.byteorder)
        start = int.from_bytes(f.read(4), byteorder=sys.byteorder)

        data = np.empty((n_examples + 1, ), dtype=int)
        data[0] = start
        data[1:] = np.fromfile(f, dtype=np.ushort)

        data = np.cumsum(data)
        return data[:-1]


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


def read_example(fd: BufferedReader,
                 ref_len: int,
                 offset: Optional[int] = None) -> Example:
    if offset is not None:
        fd.seek(offset)

    read_id, ctg, pos, n_points, q_indices_len = struct.unpack(
        '=36sHIHH', fd.read(46))
    #read_id, ctg, pos, n_points, q_indices_len, q_bases_len = struct.unpack(
    #    '=36sHIHHH', fd.read(48))

    n_bytes = 2 * n_points + 2 * q_indices_len + 3 * ref_len
    #n_bytes = 2 * n_points + 2 * q_indices_len + 3 * ref_len + q_bases_len
    data = struct.unpack(
        f'={n_points}e{q_indices_len}H{ref_len}H{ref_len}s',
        #f'={n_points}e{q_indices_len}H{ref_len}H{ref_len}s{q_bases_len}s',
        fd.read(n_bytes))
    event_len_start = n_points + q_indices_len

    signal = np.array(data[:n_points], dtype=np.half)
    q_indices = data[n_points:event_len_start]
    lengths = data[event_len_start:-1]
    bases = data[-1].decode()

    return Example(read_id.decode(), ctg, pos, signal, q_indices, lengths,
                   bases)


class ReferenceMapping:
    def __init__(self, ref_len: int, block_size: int) -> None:
        self.block_size = block_size
        self.r_pos = torch.arange(0, ref_len)  # T

    def __call__(self, e_lens: torch.Tensor) -> torch.Tensor:
        block_lengths = torch.div(e_lens,
                                  self.block_size,
                                  rounding_mode='floor')
        r_idx = torch.repeat_interleave(self.r_pos, block_lengths)  # S_B

        return r_idx


class RFTrainDataset(Dataset):
    def __init__(self, path: str, labels: str, ref_len: int,
                 block_size: int) -> None:
        super(Dataset, self).__init__()

        self.ref_len = ref_len
        self.block_size = block_size

        self.path = path
        self.fd = None
        self.ctgs = None

        self.offsets = read_offsets2(f'{path}.idx')
        # self.labels = Labels(labels)
        self.labels = np.memmap(labels, dtype=np.half)

        self.reference_mapping = ReferenceMapping(self.ref_len, block_size)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        example = read_example(self.fd, self.ref_len, self.offsets[idx])

        signal = torch.tensor(example.signal).unfold(
            -1, self.block_size, self.block_size)  # Converting to blocks

        bases = torch.tensor([ENCODING[b] for b in example.bases])
        lengths = torch.tensor(example.lengths)

        ref_mapping = self.reference_mapping(lengths)
        q_indices = torch.tensor(example.q_indices)

        #label = self.labels.get_label(example.read_id, self.ctgs[example.ctg],
        #                              example.pos)
        label = self.labels[idx]

        return signal, ref_mapping, q_indices, bases, label


class RFInferenceDataset(IterableDataset):
    def __init__(self,
                 path: str,
                 batch_size: int,
                 ref_len: int,
                 block_size: int,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None) -> None:
        super(IterableDataset, self).__init__()

        self.batch_size = batch_size
        self.ref_len = ref_len
        self.block_size = block_size

        self.mapping_encodings = ReferenceMapping(self.ref_len,
                                                  self.block_size)

        self.path = path
        self.fd = None
        with open(self.path, 'rb') as fd:
            self.ctgs = parse_ctgs(fd)

        self.offsets = read_offsets2(f'{path}.idx')
        self.start = 0 if start_idx is None else start_idx
        self.end = len(self.offsets) if end_idx is None else end_idx

    def __iter__(self):
        min_bin_idx = self.ref_len // 10
        max_bin_idx = (4 * self.ref_len) // 10
        bins = [list() for _ in range(max_bin_idx - min_bin_idx + 1)]
        stored = 0

        # Opening rockfish file
        self.fd = open(self.path, 'rb')
        self.fd.seek(self.offsets[
            self.start])  # Moving to the starting example for this worker

        for _ in range(self.start, self.end):
            example = read_example(self.fd, self.ref_len)
            bin = len(example.q_indices) // 10 - min_bin_idx
            bins[bin].append(example)
            stored += 1

            if len(bins[bin]) >= self.batch_size:  # bin is full, emit examples
                for example in bins[bin]:
                    yield self.example_to_tensor(example)

                stored -= len(bins[bin])
                bins[bin].clear()
            elif stored >= 4 * self.batch_size:  # Stored too many examples, emit some
                batch_processed = 0

                for bin in reversed(bins):
                    while len(bin) > 0:
                        example = bin.pop()
                        yield self.example_to_tensor(example)

                        batch_processed += 1
                        stored -= 1

                        if batch_processed == self.batch_size:  # Emitted whole batch
                            break

                    if batch_processed == self.batch_size:
                        break

        for bin in bins:
            for example in bin:
                yield self.example_to_tensor(example)

    def example_to_tensor(
        self, example: Example
    ) -> Tuple[str, str, int, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        signal = torch.tensor(example.signal,
                              dtype=torch.half).unfold(-1, self.block_size,
                                                       self.block_size)
        bases = torch.tensor([ENCODING.get(b, 4) for b in example.bases])
        q_indices = torch.tensor(example.q_indices)
        lengths = torch.tensor(example.lengths)

        r_pos_enc = self.mapping_encodings(lengths)

        return example.read_id, self.ctgs[
            example.ctg], example.pos, signal, bases, r_pos_enc, q_indices


def collate_fn_train(batch):
    signals, ref_mapping, q_pos_enc, bases, labels = zip(*batch)

    num_blocks = torch.tensor([len(s) for s in signals])  # B
    signals = pad_sequence(signals,
                           batch_first=True)  # [B, MAX_LEN, BLOCK_SIZE]
    ref_mapping = pad_sequence(ref_mapping, batch_first=True)  # [B, MAX_LEN]
    q_pos_enc = pad_sequence(q_pos_enc, batch_first=True)  # [B, MAX_LEN]

    return signals, ref_mapping, q_pos_enc, torch.stack(
        bases, 0), num_blocks, torch.tensor(labels)


def collate_fn_inference(batch):
    read_ids, ctgs, positions, signals, bases, ref_mapping, q_indices = zip(
        *batch)

    num_blocks = torch.tensor([len(s) for s in signals])
    signals = pad_sequence(signals, batch_first=True)  # BxMAX_LEN
    ref_mapping = pad_sequence(ref_mapping, batch_first=True)
    q_indices = pad_sequence(q_indices, batch_first=True)  # [B,MAX_LEN//5]

    return read_ids, ctgs, positions, signals, torch.stack(
        bases, 0), ref_mapping, q_indices, num_blocks


def worker_init_train_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.fd = open(dataset.path, 'rb')
    dataset.ctgs = parse_ctgs(dataset.fd)


def worker_init_rf_inference_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    total_examples = dataset.end - dataset.start
    per_worker = int(math.ceil(total_examples /
                               float(worker_info.num_workers)))

    dataset.start += worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, dataset.end)


class RFDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_data: str,
                 train_labels: str,
                 val_data: str,
                 val_labels: str,
                 train_batch_size: int = 256,
                 val_batch_size: int = 512,
                 ref_len: int = 31,
                 block_size: int = 5) -> None:
        super(RFDataModule, self).__init__()

        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.ref_len = ref_len
        self.block_size = block_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_ds = RFTrainDataset(self.train_data, self.train_labels,
                                           self.ref_len, self.block_size)
            self.val_ds = RFTrainDataset(self.val_data, self.val_labels,
                                         self.ref_len, self.block_size)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          self.train_batch_size,
                          shuffle=True,
                          collate_fn=collate_fn_train,
                          worker_init_fn=worker_init_train_fn,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          self.val_batch_size,
                          collate_fn=collate_fn_train,
                          worker_init_fn=worker_init_train_fn,
                          num_workers=4,
                          pin_memory=True)
