from os import write
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from tqdm import tqdm
import mappy

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from typing import *

from fast5 import load_read, ReadInfo
from alignment import AlignmentInfo, get_aligner, align_read
from extract import Example, extract_features, MotifPositions, build_reference_idx
from writer import BinaryWriter


def get_files(path: Path, recursive: bool = False) -> Iterator[Path]:
    if path.is_file():
        return [path]

    # Finding all input FAST5 files
    if recursive:
        files = path.glob('**/*.fast5')
    else:
        files = path.glob('*.fast5')

    return files


def get_reads(path: Path) -> Generator[Fast5Read, None, None]:
    with get_fast5_file(str(path), mode='r') as f5:
        yield from f5.get_reads()


def worker_init(aligner_: mappy.Aligner, ref_positions_: MotifPositions,
                window_: int):
    global aligner, ref_positions, window

    aligner = aligner_
    ref_positions = ref_positions_
    window = window_


def process_worker(path: Path) -> List[Example]:
    all_examples = []
    for i, read in enumerate(get_reads(path)):
        read_info = load_read(read)
        examples = extract_features(read_info, ref_positions, aligner, window)

        if examples is not None:
            all_examples.extend(examples)

        if i == 500:
            break

    return all_examples


def main(args: argparse.Namespace) -> None:
    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference)

    tqdm.write('Building reference positions for the given motif.')
    ref_positions = build_reference_idx(aligner, args.motif, args.idx)

    tqdm.write(
        f'Retrieving files from {args.source}, recursive {args.recursive}')
    files = get_files(args.source, args.recursive)

    tqdm.write('Sumbitting jobs.')
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=worker_init,
                             initargs=(aligner, ref_positions,
                                       args.window)) as pool:
        futures = [pool.submit(process_worker, p) for p in files]

        with BinaryWriter(args.dest, ref_positions.keys(),
                          2 * args.window + 1) as writer:
            writer.write_header()

            for future in tqdm(as_completed(futures)):
                writer.write_examples(future.result())

            writer.write_n_examples()


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
