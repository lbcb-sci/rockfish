import math
import sys
from typing import *

import lightning.pytorch as pl
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from rockfish.extract.extract import (MAX_BLOCKS_LEN_FACTOR,
                                      MIN_BLOCKS_LEN_FACTOR)
from rockfish.rf_format import *

ENCODING = {b: i for i, b in enumerate('ACGTN')}


def get_n_examples(idx_path: str) -> int:
    with open(idx_path, 'rb') as f:
        n_examples = int.from_bytes(f.read(4), byteorder=sys.byteorder)
        return n_examples


def read_offsets(idx_path: str) -> List[int]:
    with open(idx_path, 'rb') as f:
        n_examples = int.from_bytes(f.read(4), byteorder=sys.byteorder)
        start = int.from_bytes(f.read(4), byteorder=sys.byteorder)

        data = np.empty((n_examples + 1, ), dtype=int)
        data[0] = start
        data[1:] = np.fromfile(f, dtype=np.ushort)

        data = np.cumsum(data)
        return data[:-1]


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

    def __init__(self, path: str, labels: str, ref_len: int, block_size: int,
                 mode: str) -> None:
        super(Dataset, self).__init__()

        self.ref_len = ref_len
        self.block_size = block_size
        self.mode = mode

        self.path = path
        self.fd = None
        self.ctgs = None

        self.offsets = read_offsets(f'{path}.idx')
        # self.labels = Labels(labels)
        self.labels = np.fromfile(labels, dtype=np.half)

        # self.reference_mapping = ReferenceMapping(self.ref_len, block_size)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        example = RFExample.from_file(self.fd, self.ref_len, self.offsets[idx])

        signal = example.data.signal
        if self.mode == 'train':
            noise = np.random.randn(len(signal)).astype(
                np.half) * 0.15 * np.std(signal)
            signal = signal + noise
        signal = (signal - np.mean(signal)) / np.std(signal)

        signal = torch.tensor(signal).unfold(
            -1, self.block_size, self.block_size)  # Converting to blocks

        bases = torch.tensor([ENCODING.get(b, 4) for b in example.data.bases])
        # lengths = torch.tensor(example.data.event_lengths.astype(np.int32))

        # ref_mapping = self.reference_mapping(lengths)
        # q_indices = torch.tensor(example.data.q_indices.astype(np.int32))

        #label = self.labels.get_label(example.read_id, self.ctgs[example.ctg],
        #                              example.pos)
        label = self.labels[idx]

        singleton = True if example.data.bases.count('CG') == 1 else False

        return signal, bases, label, singleton


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

        # self.mapping_encodings = ReferenceMapping(self.ref_len, self.block_size)

        self.path = path
        self.fd = None

        self.offsets = read_offsets(f'{path}.idx')
        self.start = 0 if start_idx is None else start_idx
        self.end = len(self.offsets) if end_idx is None else end_idx

    def __iter__(self):
        min_bin_idx = int(MIN_BLOCKS_LEN_FACTOR * self.ref_len) // 10
        max_bin_idx = int(MAX_BLOCKS_LEN_FACTOR * self.ref_len) // 10
        bins = [list() for _ in range(max_bin_idx - min_bin_idx + 1)]
        stored = 0

        # Opening rockfish file
        self.fd = open(self.path, 'rb')
        self.fd.seek(self.offsets[
            self.start])  # Moving to the starting example for this worker

        for _ in range(self.start, self.end):
            example = RFExample.from_file(self.fd, self.ref_len)
            bin = (len(example.data.signal) //
                   self.block_size) // 10 - min_bin_idx
            bins[bin].append(example)
            stored += 1

            if len(bins[bin]) >= self.batch_size:  # bin is full, emit examples
                for example in bins[bin]:
                    yield self.example_to_tensor(example)

                stored -= len(bins[bin])
                bins[bin].clear()
            elif stored >= 4 * self.batch_size:  # Stored too many examples, emit some
                batch_processed = 0

                for bin in bins:
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
            self,
            example: RFExample) -> Tuple[str, int, torch.Tensor, torch.Tensor]:
        signal = torch.tensor(example.data.signal,
                              dtype=torch.half).unfold(-1, self.block_size,
                                                       self.block_size)
        bases = torch.tensor([ENCODING.get(b, 4) for b in example.data.bases])

        return example.header.read_id, example.header.pos, signal, bases


def collate_fn_train(batch):
    signals, bases, labels, singleton = zip(*batch)

    num_blocks = torch.tensor([len(s) for s in signals])  # B
    signals = pad_sequence(signals,
                           batch_first=True)  # [B, MAX_LEN, BLOCK_SIZE]
    # ref_mapping = pad_sequence(ref_mapping, batch_first=True)  # [B, MAX_LEN]
    # q_pos_enc = pad_sequence(q_pos_enc, batch_first=True)  # [B, MAX_LEN]

    return signals, torch.stack(
        bases, 0), num_blocks, torch.tensor(labels), torch.tensor(singleton)


def collate_fn_inference(batch):
    read_ids, positions, signals, bases = zip(*batch)

    num_blocks = torch.tensor([len(s) for s in signals])
    signals = pad_sequence(signals, batch_first=True)  # BxMAX_LEN

    return read_ids, positions, signals, torch.stack(bases, 0), num_blocks


def worker_init_train_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.fd = open(dataset.path, 'rb')
    # dataset.ctgs = parse_ctgs(dataset.fd)


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
                                           self.ref_len, self.block_size,
                                           'train')

            self.val_ds = RFTrainDataset(self.val_data, self.val_labels,
                                         self.ref_len, self.block_size, 'val')

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
