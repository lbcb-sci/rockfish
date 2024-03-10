from __future__ import annotations

import mappy
import pysam
from tqdm import tqdm

from functools import partial
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import argparse

from typing import *

MotifPositions = Dict[str, Tuple[Set[int], Set[int]]]


@dataclass
class Alignment:
    reference_name: str
    reference_start: int
    reference_end: int
    is_reverse: bool
    query_name: str
    query_sequnece: str
    cigartuples: List[Tuple[int, int]]

    @classmethod
    def from_aligned_segment(cls, record: pysam.AlignedSegment) -> Alignment:
        return cls(record.reference_name, record.reference_start,
                   record.reference_end, record.is_reverse, record.query_name,
                   record.query_sequence, record.cigartuples)


TABLE_BYTES = bytes.maketrans(b"ACTG", b"TGAC")


def reverse_complement_bytes(seq: str):
    return seq.translate(TABLE_BYTES)[::-1]


def build_index_for_ctg(record, motif: str,
                        rel_idx: str) -> Tuple[Set[int], Set[int]]:
    sequence = record[1]
    fwd_pos = {m.start() + rel_idx for m in re.finditer(motif, sequence, re.I)}

    rev_comp = mappy.revcomp(sequence)
    seq_len = len(rev_comp)

    def pos_for_rev(i: int) -> int:
        return seq_len - (i + rel_idx) - 1

    rev_pos = {
        pos_for_rev(m.start())
        for m in re.finditer(motif, rev_comp, re.I)
    }

    return record[0], (fwd_pos, rev_pos)


def build_reference_idx2(reference: str, motif: str, rel_idx: int,
                         n_workers: int) -> MotifPositions:

    futures = []
    with ProcessPoolExecutor(n_workers) as pool:
        idx_func = partial(build_index_for_ctg, motif=motif, rel_idx=rel_idx)
        for record in mappy.fastx_read(reference):
            future = pool.submit(idx_func, record)
            futures.append(future)

    positions = {}
    for future in as_completed(futures):
        c, p = future.result()
        positions[c] = p
    return positions


def process_record(motif: str, rel_idx: int, positions: MotifPositions,
                   in_queue: mp.Queue[Alignment], out_queue: mp.Queue):

    while (record := in_queue.get()) is not None:
        seq = reverse_complement_bytes(
            record.query_sequnece
        ) if record.is_reverse else record.query_sequnece
        query_pos = {
            m.start() + rel_idx
            for m in re.finditer(motif, seq, re.I)
        }

        qpos = 0
        if record.is_reverse:
            rpos = record.reference_end - 1
            cigar = reversed(record.cigartuples)
            ref_pos = positions[record.reference_name][1]
        else:
            rpos = record.reference_start
            cigar = record.cigartuples
            ref_pos = positions[record.reference_name][0]

        mappings = set()
        for op, l in cigar:
            if op == 0 or op == 7 or op == 8:
                for _ in range(l):
                    mappings.add((rpos, qpos))
                    rpos = rpos - 1 if record.is_reverse else rpos + 1
                    qpos += 1
            elif op == 1 or op == 4:
                qpos += l
            elif op == 2:
                rpos = rpos - l if record.is_reverse else rpos + l
            else:
                raise ValueError('Invalid cigar op.')
        assert qpos == len(seq)

        mappings = [(rpos, qpos) for rpos, qpos in mappings
                    if rpos in ref_pos and qpos in query_pos]

        out_queue.put((record.query_name, record.reference_name, mappings))
    out_queue.put(None)


def bam_read_worker(path, minq, queue, n_workers):
    with pysam.AlignmentFile(path, mode='rb', check_sq=False, threads=16) as bam:
        for record in bam:
            if record.is_secondary or record.is_unmapped or record.is_qcfail or record.is_supplementary:
                continue
            if record.mapping_quality < minq:
                continue

            record = Alignment.from_aligned_segment(record)
            queue.put(record)

    for _ in range(n_workers):
        queue.put(None)


def main(args):
    ref_positions = build_reference_idx2(args.reference, args.motif, args.idx,
                                         args.workers)

    in_queue = mp.Queue(10000)
    out_queue = mp.Queue(10000)
    bam_reader = mp.Process(target=bam_read_worker,
                            args=(args.bam, args.minq, in_queue, args.workers))
    bam_reader.start()

    for _ in range(args.workers):
        p = mp.Process(target=process_record,
                       args=(args.motif, args.idx, ref_positions, in_queue,
                             out_queue))
        p.start()

    pbar, finished = tqdm(), 0
    while True:
        result = out_queue.get()
        if result is None:
            finished += 1
            if finished == args.workers:
                break
        else:
            qname, rname, mappings = result
            for rpos, qpos in mappings:
                print(qname, qpos, rpos, rname, sep='\t')
            pbar.update()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('bam', type=str)
    parser.add_argument('reference', type=str)
    parser.add_argument('--motif', type=str, default='CG')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--minq', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
