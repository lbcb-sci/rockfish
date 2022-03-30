from tqdm import tqdm

from dataclasses import dataclass
import sys
from io import BufferedWriter
import struct
from collections import OrderedDict
from itertools import count
import argparse

from typing import *

from rockfish.rf_format import *


def map_header(src: List[str]) -> Tuple[Dict[str, int], List[RFHeader]]:
    ctg_encoding = OrderedDict()
    headers = []

    for rf_file in src:
        with open(rf_file, 'rb') as rf_src:
            header = RFHeader.parse_header(rf_src)
            headers.append(header)

            for ctg in header.ctgs:
                if ctg not in ctg_encoding:
                    ctg_encoding[ctg] = len(ctg_encoding)

    return ctg_encoding, headers


def write_header(dest: BufferedWriter, ctg_encoding: Dict[str, int],
                 n_examples: int) -> int:
    # Writing ctgs + placeholder for number of examples
    header = RFHeader([c for c, _ in ctg_encoding.items()], n_examples)
    dest.write(header.to_bytes())

    return header.size() - 4


def merge(src: List[str], dest: str, seq_len: int) -> None:
    tqdm.write('Reading source headers.')

    ctg_encoding, headers = map_header(src)
    n_examples = sum([h.n_examples for h in headers])
    tqdm.write(f'Total {n_examples} examples found.')

    with open(dest, 'wb') as rf_dest:
        write_header(rf_dest, ctg_encoding, n_examples)
        tqdm.write('Destination header written.\nWriting examples...')

        for rf_file, header in zip(tqdm(src), headers):
            with open(rf_file, 'rb') as rf_src:
                rf_src.seek(header.size())

                for _ in range(header.n_examples):
                    example_header = RFExampleHeader.parse_bytes(
                        rf_src.read(EXAMPLE_HEADER_STRUCT.size))
                    data = rf_src.read(example_header.example_len(seq_len))

                    rf_dest.write(example_header.to_bytes() + data)

                # assert EOF
                if rf_src.read():
                    tqdm.write(
                        f'ERROR: {rf_file} EOF expected but data was read.')
                    sys.exit(1)

    tqdm.write('Processing finished')


def add_merge_arguments(parser: argparse.ArgumentParser) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('src', type=str, nargs='+')
    parser.add_argument('dest', type=str)
    parser.add_argument('-l', '--seq_len', type=int, default=31)


def main(args):
    merge(args.src, args.dest, args.seq_len)
