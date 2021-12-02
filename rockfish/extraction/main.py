from os import write
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from tqdm import tqdm

from pathlib import Path
import argparse

from typing import *

from fast5 import load_read, ReadInfo
from alignment import AlignmentInfo, get_aligner, align_read
from extract import extract_features, MotifPositions, build_reference_idx
from writer import BinaryWriter


def get_files(path: Path, recursive: bool = False) -> List[Path]:
    if path.is_file():
        return [path]

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return list(files)


def get_reads(path: Path) -> Generator[Fast5Read, None, None]:
    with get_fast5_file(str(path), mode='r') as f5:
        yield from f5.get_reads()


def main(args: argparse.Namespace) -> None:
    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference)
    # ref_positions = build_reference_idx(aligner, args.motif, args.idx)

    tqdm.write(
        f'Retrieving files from {args.source}, recursive {args.recursive}')
    files = get_files(args.source, args.recursive)

    for f in files:
        for r in get_reads(f):
            print(r.read_id)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('dest', type=Path)

    parser.add_argument('--motif', type=str, default='CG')
    parser.add_argument('--idx', type=int, default=0)

    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('--window', type=int, default=15)

    parser.add_argument('-t', '--workers', type=int, default=1)

    parser.add_argument('--info_file', type=str, default='info.txt')

    parser.add_argument('--read_ids', type=Path, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    main(args)
