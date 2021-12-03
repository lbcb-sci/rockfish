import mappy
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import sys

from typing import *


@dataclass
class AlignmentInfo:
    ctg: str
    r_start: int
    r_end: int
    fwd_strand: bool
    ref_to_query: np.ndarray


def align_read(query: str, aligner: mappy.Aligner, mapq_filter: bool,
               read_id: str) -> Optional[AlignmentInfo]:
    alignments = list(aligner.map(query))
    if not alignments:
        return None

    alignment = alignments[0]
    if mapq_filter and alignment.mapq < 60:
        return None

    ref_len = alignment.r_en - alignment.r_st
    cigar = alignment.cigar if alignment.strand == 1 else reversed(
        alignment.cigar)
    rpos, qpos = 0, alignment.q_st  # if alignment.strand == 1 else len(query) - alignment.q_en

    ref_to_query = np.empty((ref_len + 1, ), dtype=int)
    for l, op in cigar:
        if op == 0 or op == 7 or op == 8:  # Match or mismatch
            for i in range(l):
                ref_to_query[rpos + i] = qpos + i
            rpos += l
            qpos += l
        elif op == 1:  # Insertion
            qpos += l
        elif op == 2:
            for i in range(l):
                ref_to_query[rpos + i] = qpos
            rpos += l
    ref_to_query[rpos] = qpos  # Add the last one (excluded end)

    if ref_len != rpos:
        print(
            f'Warning: Mismatch between reference ({ref_len}) and cigar ({rpos}) length in query: {read_id}',
            file=sys.stderr)

    return AlignmentInfo(alignment.ctg, alignment.r_st, alignment.r_en,
                         alignment.strand == 1, ref_to_query)


def get_aligner(reference_path: Path) -> mappy.Aligner:
    return mappy.Aligner(str(reference_path), preset='map-ont', best_n=1)
