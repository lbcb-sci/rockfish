import argparse
import os
import random
import sys
from collections import OrderedDict
from io import BufferedWriter
from typing import *

from tqdm import tqdm

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


def merge(src: List[str], dest: str, n_samples: int, seq_len: int,
          delete_src: bool, seed: Optional[int]) -> None:
    random.seed(seed)
    if seed is not None:
        tqdm.write(f'Random seed {seed}')
    else:
        tqdm.write(f'Random seed is not set.')

    tqdm.write('Reading source headers.')

    ctg_encoding, headers = map_header(src)
    n_examples = sum([h.n_examples for h in headers])
    tqdm.write(f'Total {n_examples} examples found.')

    sampled_indices = set(random.sample(range(n_examples), k=n_samples))
    tqdm.write(f'Sampled {n_samples} examples.')

    with open(dest, 'wb') as rf_dest:
        write_header(rf_dest, ctg_encoding, n_samples)
        tqdm.write('Destination header written.\nWriting examples...')

        i = 0
        for rf_file, header in zip(tqdm(src), headers):
            with open(rf_file, 'rb') as rf_src:
                rf_src.seek(header.size())

                for _ in range(header.n_examples):
                    example_header = RFExampleHeader.parse_bytes(
                        rf_src.read(EXAMPLE_HEADER_STRUCT.size))
                    data = rf_src.read(example_header.example_len(seq_len))

                    if i not in sampled_indices:
                        i += 1
                        continue
                    i += 1

                    ctg = header.ctgs[example_header.ctg_id]
                    example_header.ctg_id = ctg_encoding[ctg]

                    rf_dest.write(example_header.to_bytes() + data)

                # assert EOF
                if rf_src.read():
                    tqdm.write(
                        f'ERROR: {rf_file} EOF expected but data was read.')
                    sys.exit(1)

    tqdm.write('Processing finished')

    if delete_src:
        tqdm.write('Deleting source files...')
        for path in src:
            os.remove(path)

    tqdm.write('Processing finished')


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('src', type=str, nargs='+')
    parser.add_argument('dest', type=str)
    parser.add_argument('-n', '--n_samples', type=int, required=True)
    parser.add_argument('-l', '--seq_len', type=int, default=31)
    parser.add_argument('-d', '--delete_src', action='store_true')
    parser.add_argument('--seed', type=int, default=None)


def main(args):
    merge(args.src, args.dest, args.n_samples, args.seq_len, args.delete_src,
          args.seed)
