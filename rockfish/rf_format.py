from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BufferedReader
from struct import Struct
from typing import *

import numpy as np


@dataclass
class RFHeader:
    ctgs: List[str]
    n_examples: int

    @classmethod
    def parse_header(cls, fd: BufferedReader) -> RFHeader:
        fd.seek(0)

        n_ctgs = int.from_bytes(fd.read(2), byteorder=sys.byteorder)
        ctgs = []
        for _ in range(n_ctgs):
            ctg_name_len = int.from_bytes(fd.read(1), byteorder=sys.byteorder)
            ctg_name = fd.read(ctg_name_len).decode()
            ctgs.append(ctg_name)

        n_examples = int.from_bytes(fd.read(4), byteorder=sys.byteorder)

        return cls(ctgs, n_examples)

    def to_bytes(self) -> bytes:
        data = len(self.ctgs).to_bytes(2, byteorder=sys.byteorder)

        for ctg in self.ctgs:
            data += len(ctg).to_bytes(1, byteorder=sys.byteorder)
            data += str.encode(ctg)

        data += self.n_examples.to_bytes(4, byteorder=sys.byteorder)

        return data

    def size(self) -> int:
        return 2 + len(self.ctgs) + sum([len(c) for c in self.ctgs]) + 4


EXAMPLE_HEADER_STRUCT = Struct('=36sHIHH')


@dataclass
class RFExampleHeader:
    read_id: str
    ctg_id: int
    pos: int
    n_points: int
    q_indices_len: int

    @classmethod
    def parse_bytes(cls, data: bytes) -> RFExampleHeader:
        read_id, ctg_id, pos, n_points, q_indices_len = EXAMPLE_HEADER_STRUCT.unpack(
            data)

        return cls(read_id.decode(), ctg_id, pos, n_points, q_indices_len)

    def to_bytes(self) -> bytes:
        return EXAMPLE_HEADER_STRUCT.pack(str.encode(self.read_id), self.ctg_id,
                                          self.pos, self.n_points,
                                          self.q_indices_len)

    def example_len(self, seq_len: int) -> int:
        return 2 * self.n_points + 2 * self.q_indices_len + 3 * seq_len + 2 * seq_len  # 2*S For current diff


DataArray = Union[List, np.ndarray]


def convert_array(data: DataArray, dtype: np.dtype) -> np.ndarray:
    if isinstance(data, list):
        return np.array(data, dtype=dtype)
    elif isinstance(data, np.ndarray):
        if data.dtype == dtype:
            return data

        return data.astype(dtype)
    else:
        raise ValueError('Invalid data type')


@dataclass
class RFExampleData:
    signal: np.ndarray
    q_indices: np.ndarray
    event_lengths: np.ndarray
    bases: str
    diff_means: np.ndarray

    def __init__(self, signal: DataArray, q_indices: DataArray,
                 event_lengths: DataArray, bases: str,
                 diff_menas: DataArray) -> None:
        super().__init__()

        self.signal = convert_array(signal, np.half)
        self.q_indices = convert_array(q_indices, np.ushort)
        self.event_lengths = convert_array(event_lengths, np.ushort)
        self.bases = bases
        self.diff_means = convert_array(diff_menas, np.half)

    @classmethod
    def parse_bytes(cls, data: bytes, header: RFExampleHeader,
                    seq_len: int) -> RFExampleData:
        start, end = 0, 2 * header.n_points
        signal = np.frombuffer(data[start:end], np.half)

        start, end = end, end + 2 * header.q_indices_len
        q_indices = np.frombuffer(data[start:end], np.ushort)

        start, end = end, end + 2 * seq_len
        event_lengths = np.frombuffer(data[start:end], np.ushort)

        start, end = end, end + seq_len
        bases = data[start:end].decode()

        start, end = end, end + 2 * seq_len
        diff_means = np.frombuffer(data[start:end], np.half)

        return cls(signal, q_indices, event_lengths, bases, diff_means)

    def to_bytes(self) -> bytes:
        return self.signal.tobytes() + self.q_indices.tobytes(
        ) + self.event_lengths.tobytes() + str.encode(
            self.bases) + self.diff_means.tobytes()


@dataclass
class RFExample:
    header: RFExampleHeader
    data: RFExampleData

    @classmethod
    def from_file(cls,
                  fd: BufferedReader,
                  seq_len: int,
                  offset: Optional[int] = None) -> RFExample:
        if offset is not None:
            fd.seek(offset)

        header = RFExampleHeader.parse_bytes(fd.read(
            EXAMPLE_HEADER_STRUCT.size))
        data = RFExampleData.parse_bytes(fd.read(header.example_len(seq_len)),
                                         header, seq_len)

        return cls(header, data)


class DictLabels:

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
        elif (read_id, ctg, pos) in self.label_for_read_pos:
            return self.label_for_read_pos[(read_id, ctg, pos)]

        raise KeyError('Label for the given example is not provided')
