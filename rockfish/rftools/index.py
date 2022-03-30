from __future__ import annotations

from tqdm import tqdm

import sys
from io import BufferedReader
from pathlib import Path
from dataclasses import dataclass
import argparse

from rf_format import *

from typing import *


@dataclass
class RFIndex:
    n_examples: int
    offsets: np.ndarray

    @classmethod
    def parse(cls, path: str) -> RFIndex:
        with open(path, 'rb') as index:
            n_examples = int.from_bytes(index.read(4), sys.byteorder)
            start = int.from_bytes(index.read(4), sys.byteorder)

            lens = [0]
            for _ in range(n_examples):
                l = int.from_bytes(index.read(2), sys.byteorder)
                lens.append(l)

        offsets = start + np.cumsum(lens)
        return cls(n_examples, offsets)

    def __getitem__(self, idx: int) -> int:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(
                f'Total {self.n_examples} examples. Index {idx} is out of bounds.'
            )

        return self.offsets[idx]


def get_n_examples(data: BufferedReader) -> int:
    data.seek(0)

    n_ctgs = int.from_bytes(data.read(2), byteorder=sys.byteorder)
    for _ in range(n_ctgs):
        ctg_name_len = int.from_bytes(data.read(1), byteorder=sys.byteorder)
        data.read(ctg_name_len)

    return int.from_bytes(data.read(4), byteorder=sys.byteorder, signed=False)


def index(rf_path: Path, index_path: Optional[Path], seq_len: int) -> None:
    if index_path is None:
        index_path = rf_path.parent / (rf_path.name + '.idx')

    with rf_path.open('rb') as data:
        header = RFHeader.parse_header(data)

        with index_path.open('wb') as index_fd:
            index_fd.write(
                header.n_examples.to_bytes(4, sys.byteorder, signed=False))
            index_fd.write(header.size().to_bytes(4,
                                                  sys.byteorder,
                                                  signed=False))

            tqdm.write('Writing indices...')
            for i in tqdm(range(header.n_examples)):
                example_header = RFExampleHeader.parse_bytes(
                    data.read(EXAMPLE_HEADER_STRUCT.size))
                data_bytes_len = example_header.example_len(seq_len)
                data.read(data_bytes_len)

                total = EXAMPLE_HEADER_STRUCT.size + data_bytes_len
                index_fd.write(total.to_bytes(2, sys.byteorder))

    tqdm.write(f'Finished index generation.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=Path)
    parser.add_argument('-i',
                        '--index_path',
                        type=Optional[Path],
                        default=None)
    parser.add_argument('-s', '--seq_len', type=int, default=31)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    index(args.data_path, args.index_path, args.seq_len)
