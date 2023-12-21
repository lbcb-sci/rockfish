from pathlib import Path
import sys
from collections import OrderedDict

from typing import List, Set

from rockfish.rf_format import *
from .extract import Example


class BinaryWriter:
    def __init__(self, path: Path, ref_names: Set[str], seq_len: int) -> None:
        self.path = path
        self.S = seq_len

        self.ref_ids = OrderedDict([(c, i) for i, c in enumerate(ref_names)])
        self.n_examples = 0

    def __enter__(self):
        self.fd = self.path.open('wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fd.close()

    def write_example(self, example: Example) -> bytes:
        header = RFExampleHeader(example.read_id, self.ref_ids[example.ctg],
                                 example.pos, len(example.signal),
                                 len(example.q_indices)).to_bytes()
        data = RFExampleData(example.signal, example.q_indices,
                             example.event_length, example.bases).to_bytes()

        return header + data

    def write_examples(self, examples: List[Example]) -> None:
        self.fd.write(b''.join([self.write_example(e) for e in examples]))
        self.n_examples += len(examples)

    def write_header(self) -> None:
        data = RFHeader(list(self.ref_ids.keys()), 0).to_bytes()
        self.header_offset = len(data)

        self.fd.write(data)

    def write_n_examples(self) -> None:
        self.fd.seek(self.header_offset - 4)
        data = self.n_examples.to_bytes(4, byteorder=sys.byteorder)
        self.fd.write(data)
