from pathlib import Path
from collections import defaultdict
from itertools import count
import struct

from typing import List, Set

from extract import Example

class BinaryWriter:
    def __init__(self, path: Path, ref_names: Set[str]) -> None:
        self.path = path

        self.ref_ids = {n: i for i, n in enumerate(ref_names)}
        self.n_examples = 0

    def __enter__(self):
        self.fd = self.path.open('wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd.close()

    def write_example(self, example: Example) -> None:
        ref_id = self.ref_ids[example.ctg]
        q_name_len = len(example.read_id)
        n_points = len(example.signal)
        pos = example.pos

        q_name = str.encode(example.read_id)
        seq = str.encode(example.bases)

        data = struct.pack(f'=IBII{q_name_len}s{n_points}e31s', 
                           ref_id, q_name_len, n_points, pos, q_name,
                           *example.signal, seq)

        self.fd.write(data)
        self.n_examples += 1

    def write_examples(self, examples: List[Example]) -> None:
        for example in examples:
            self.write_example(example)

    def write_header(self) -> None:
        n_refs = len(self.ref_ids)
        data = struct.pack('=I', n_refs)

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