from ctypes import alignment
from os import write
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from tqdm import tqdm
import mappy
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from itertools import count
from pathlib import Path
import re
import multiprocessing as mp
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


def get_reads(path: Path) -> Iterator[Fast5Read]:
    with get_fast5_file(str(path), mode='r') as f5:
        for read in f5.get_reads():
            yield read


def loader_worker(files: List[Path], read_queue: mp.Queue,
                  n_reads: mp.Value) -> None:
    for file in files:
        for r in get_reads(file):
            read_info = load_read(r)

            read_queue.put(read_info)
            #with n_reads.get_lock():
               # n_reads.value += 1


def process_worker(ref_positions: MotifPositions, aligner: mappy.Aligner,
                   window: int, read_queue: mp.Queue, 
                   writer_queue: mp.Queue, n_processed: mp.Value) -> None:
    while (read_info := read_queue.get()) is not None:
        examples = extract_features(read_info, ref_positions, aligner, window)
        if examples is not None:
            writer_queue.put(examples)
        
        with n_processed.get_lock():
            n_processed.value += 1


"""def writer_worker(path: Path, write_queue: mp.Queue) -> None:
    ref_ids = defaultdict(lambda c=count(): next(c))
    with path.open('w') as writer:
        while (examples := write_queue.get()) is not None:
            for example in examples:
                ex = (
                    f'{example.read_id}\t'  # read_id
                    f'{ref_ids[example.ctg]}\t'  # ctg
                    f'{example.pos}\t'  # position
                    f'{",".join(str(v) for v in example.signal)}\t'  # signal
                    f'{example.bases}\n'  # ref abses
                )
                writer.write(ex)"""
def writer_worker(path: Path, refs: Set[str], write_queue: mp.Queue) -> None:
    with BinaryWriter(path, refs) as writer:
        writer.write_header()

        while (examples := write_queue.get()) is not None:
            writer.write_examples(examples)

        writer.write_n_examples()


def main(args: argparse.Namespace) -> None:
    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference)
    ref_positions = build_reference_idx(aligner, args.motif, args.idx)

    tqdm.write(f'Retrieving files from {args.source}, recursive {args.recursive}')
    files = get_files(args.source, args.recursive)

    read_queue = mp.Queue(maxsize=10_000)
    writer_queue = mp.Queue(maxsize=10_000)
    n_reads = mp.Value('i', 0)
    n_processed = mp.Value('i', 0)

    loader = mp.Process(target=loader_worker, args=(files, read_queue, n_reads),
                        daemon=True)
    loader.start()

    processors = []
    for _ in range(args.workers):
        p = mp.Process(target=process_worker, 
                       args=(ref_positions, aligner, args.window, read_queue, 
                             writer_queue, n_processed), daemon=True)
        p.start()
        processors.append(p)

    writer = mp.Process(target=writer_worker, 
                        args=(args.dest, ref_positions.keys(), writer_queue), 
                        daemon=True)
    writer.start()

    with tqdm() as pbar:
        while loader.is_alive():
            # pbar.total = n_reads.value

            curr_processed = n_processed.value
            pbar.update(curr_processed - pbar.n)        

    for _ in processors:
        read_queue.put(None)
    for p in processors:
        p.join()
    tqdm.write('Processing finished.')

    writer_queue.put(None)
    writer.join()
    tqdm.write('All examples written.')
    

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
