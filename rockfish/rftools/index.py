from tqdm import tqdm

import sys
from io import BufferedReader
import struct
from pathlib import Path
import argparse

from typing import *


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
        n_examples = get_n_examples(
            data)  # Get number of examples and advance pointer

        example_idx = data.tell()
        with index_path.open('wb') as index_fd:
            index_fd.write(n_examples.to_bytes(4, sys.byteorder, signed=False))

            tqdm.write('Writing indices...')
            for i in tqdm(range(n_examples)):
                index_fd.write(
                    example_idx.to_bytes(8, sys.byteorder, signed=False))

                _, _, _, n_points, q_indices_len, q_bases_len = struct.unpack(
                    '=36sHIHHH', data.read(48))
                n_bytes = 2 * n_points + 2 * q_indices_len + 3 * seq_len + q_bases_len
                data.read(n_bytes)

                example_idx += 48 + n_bytes

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
