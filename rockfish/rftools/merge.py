from tqdm import tqdm

from dataclasses import dataclass
import sys
from io import BufferedReader, BufferedWriter
import struct
from collections import defaultdict
from itertools import count
import argparse

from typing import *


@dataclass
class RFHeader:
    ctgs: List[int]
    n_examples: int
    start_offset: int


def parse_ctgs(fd: BufferedReader) -> List[str]:
    fd.seek(0)

    n_ctgs = int.from_bytes(fd.read(2), byteorder=sys.byteorder)
    ctgs = []
    for _ in range(n_ctgs):
        ctg_name_len = int.from_bytes(fd.read(1), byteorder=sys.byteorder)
        ctg_name = struct.unpack(f'={ctg_name_len}s', fd.read(ctg_name_len))[0]
        ctgs.append(ctg_name.decode())

    return ctgs


def map_header(src: List[str]) -> Tuple[Dict[str, int], Dict[str, RFHeader]]:
    ctg_encoding = defaultdict(lambda c=count(): next(c))
    header_info = []

    for rf_file in src:
        with open(rf_file, 'rb') as rf_src:
            c_map = [ctg_encoding[c] for c in parse_ctgs(rf_src)]
            n_examples = int.from_bytes(rf_src.read(4),
                                        byteorder=sys.byteorder,
                                        signed=False)
            start_offset = rf_src.tell()

        header_info.append(RFHeader(c_map, n_examples, start_offset))

    return ctg_encoding, header_info


def write_header(dest: BufferedWriter, ctg_encoding: Dict[str, int],
                 n_examples: int) -> int:
    # Writing ctgs + placeholder for number of examples
    n_refs = len(ctg_encoding)
    data = struct.pack('=H', n_refs)

    for ref_name, _ in ctg_encoding.items():
        ref_len = len(ref_name)
        data += struct.pack(f'=B{ref_len}s', ref_len, str.encode(ref_name))

    n_examples_offset = len(data)
    data += struct.pack('=I', n_examples)
    dest.write(data)

    return n_examples_offset


def merge(src: List[str], dest: str, seq_len: int) -> None:
    tqdm.write('Reading source headers.')

    ctg_encoding, header_info = map_header(src)
    n_examples = sum([h.n_examples for h in header_info])
    tqdm.write(f'Total {n_examples} examples found.')

    with open(dest, 'wb') as rf_dest:
        write_header(rf_dest, ctg_encoding, n_examples)
        tqdm.write('Destination header written.\nWriting examples...')

        for rf_file, header in zip(tqdm(src), header_info):
            with open(rf_file, 'rb') as rf_src:
                rf_src.seek(header.start_offset)

                for _ in range(header.n_examples):
                    example_info = rf_src.read(46)
                    _, _, _, n_points, q_indices_len = struct.unpack(
                        '=36sHIHH', example_info)
                    n_bytes = 2 * n_points + 2 * q_indices_len + 3 * seq_len
                    example_data = rf_src.read(n_bytes)

                    rf_dest.write(example_info + example_data)

                # assert EOF
                if rf_src.read():
                    tqdm.write(
                        f'ERROR: {rf_file} EOF expected but data was read.')
                    sys.exit(1)

    tqdm.write('Processing finished')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('src', type=str, nargs='+')
    parser.add_argument('dest', type=str)
    parser.add_argument('-l', '--seq_len', type=int, default=31)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    merge(args.src, args.dest, args.seq_len)
