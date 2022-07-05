from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from tqdm import tqdm
import mappy

from pathlib import Path
from collections import Counter
import multiprocessing as mp
import argparse

from typing import *

from rockfish.rftools.merge import merge
from .fast5 import load_read
from .alignment import get_aligner, AlignmentInfo
from .extract import extract_features, MotifPositions, build_reference_idx2
from .writer import BinaryWriter

import traceback


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


def process_worker(aligner: mappy.Aligner, ref_positions: MotifPositions,
                   window: int, mapq_filter: int, unique_aln: bool,
                   dest_path: Path, in_queue: mp.Queue,
                   out_queue: mp.Queue) -> None:
    with BinaryWriter(dest_path, ref_positions.keys(),
                      2 * window + 1) as writer:
        writer.write_header()

        buffer = mappy.ThreadBuffer()
        while (path := in_queue.get()) is not None:
            status_count = Counter({e.name: 0 for e in AlignmentInfo})
            for read in get_reads(path):
                try:
                    read_info = load_read(read)
                    status, examples = extract_features(read_info,
                                                        ref_positions, aligner,
                                                        buffer, window,
                                                        mapq_filter, unique_aln)

                    if examples is not None:
                        writer.write_examples(examples)

                    status_count[status.name] += 1  # Update status
                except Exception as e:
                    tqdm.write(
                        f'Exception occured for read: {read_info.read_id} in file {path}'
                    )
                    tqdm.write(traceback.format_exc())

            out_queue.put(status_count)

        writer.write_n_examples()


def extract(args: argparse.Namespace) -> None:
    tqdm.write(
        f'Retrieving files from {args.source}, recursive {args.recursive}')
    files = list(get_files(args.source, args.recursive))

    in_queue = mp.Queue()
    n_workers = min(args.workers, len(files))

    for p in files:
        in_queue.put(p)
    for _ in range(n_workers):
        in_queue.put(None)

    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference, args.workers)

    tqdm.write('Building reference positions for the given motif.')
    ref_positions = build_reference_idx2(aligner, args.motif, args.idx,
                                         args.workers)

    out_queue = mp.Queue()

    workers = [None] * n_workers
    writers_path = [None] * n_workers
    for i in range(n_workers):
        writers_path[i] = args.dest.parent / (args.dest.name + f'.{i}.tmp')
        workers[i] = mp.Process(target=process_worker,
                                args=(aligner, ref_positions, args.window,
                                      args.mapq_filter, args.unique,
                                      writers_path[i], in_queue, out_queue),
                                daemon=True)
        workers[i].start()

    tqdm.write('Processing started.')

    pbar = tqdm(total=len(files))
    status_count = Counter({e.name: 0 for e in AlignmentInfo})
    while pbar.n < len(files):
        status = out_queue.get()
        status_count += status

        pbar.set_postfix(status_count, refresh=False)
        pbar.update()

    for w in workers:  # All workers should finish soon
        w.join()

    merge(writers_path, args.dest, 2 * args.window + 1, args.delete_src)
    #for p in writers_path:  # Removing tmp files
    #    p.unlink()


def add_extract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('dest', type=Path)

    parser.add_argument('--motif', type=str, default='CG')
    parser.add_argument('--idx', type=int, default=0)

    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('-w', '--window', type=int, default=15)
    parser.add_argument('-q', '--mapq_filter', type=int, default=0)
    parser.add_argument('-u', '--unique', action='store_true')

    parser.add_argument('-t', '--workers', type=int, default=1)

    parser.add_argument('--info_file', type=str, default='info.txt')

    parser.add_argument('--read_ids', type=Path, default=None)

    parser.add_argument('--delete_src', action='store_true')
