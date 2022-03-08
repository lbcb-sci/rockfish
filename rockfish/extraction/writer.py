from pathlib import Path
from collections import defaultdict
from itertools import count
import struct

from typing import List, Set

from extract import Example


class BinaryWriter:

    def __init__(self, path: Path, ref_names: Set[str], seq_len: int) -> None:
        self.path = path
        self.S = seq_len

        self.ref_ids = {n: i for i, n in enumerate(ref_names)}
        self.n_examples = 0

    def __enter__(self):
        self.fd = self.path.open('wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd.close()

    def write_example(self, example: Example) -> bytes:
        ref_id = self.ref_ids[example.ctg]
        n_points = len(example.signal)
        q_indices_len = len(example.q_indices)

        data = struct.pack(
            f'=36sHIHH{n_points}e{q_indices_len}H{self.S}H{self.S}s',
            str.encode(example.read_id), ref_id, example.pos, n_points,
            q_indices_len, *example.signal, *example.q_indices,
            *example.event_length, str.encode(example.bases))
        return data

    def write_examples(self, examples: List[Example]) -> None:
        self.fd.write(b''.join([self.write_example(e) for e in examples]))
        self.n_examples += len(examples)

    def write_header(self) -> None:
        n_refs = len(self.ref_ids)
        data = struct.pack('=H', n_refs)

        for ref_name, _ in self.ref_ids.items():
            ref_len = len(ref_name)
            data += struct.pack(f'=B{ref_len}s', ref_len, str.encode(ref_name))

        self.header_offset = len(data)
        data += struct.pack('=I', 0)  # Placeholder for n_examples

        self.fd.write(data)

    def write_n_examples(self) -> None:
        self.fd.seek(self.header_offset)
        data = struct.pack('=I', self.n_examples)
        self.fd.write(data)
