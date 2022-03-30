import mappy
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

from typing import *


@dataclass
class AlignmentData:
    ctg: str
    r_start: int
    r_end: int
    fwd_strand: bool
    ref_to_query: np.ndarray


class AlignmentInfo(Enum):
    SUCCESS = 0,
    NO_ALIGNMENT = 1,
    UNIQUE_FAIL = 2,
    MAPQ_FAIL = 3


def align_read(query: str, aligner: mappy.Aligner, mapq_filter: int,
               unique_filter: bool,
               read_id: str) -> Tuple[AlignmentInfo, Optional[AlignmentData]]:
    alignments = list(aligner.map(query))
    if not alignments:
        return AlignmentInfo.NO_ALIGNMENT, None
    if unique_filter and len(alignments) > 1:
        return AlignmentInfo.UNIQUE_FAIL, None

    alignment = alignments[0]
    if alignment.mapq < mapq_filter:
        return AlignmentInfo.MAPQ_FAIL, None

    ref_len = alignment.r_en - alignment.r_st
    cigar = alignment.cigar if alignment.strand == 1 else reversed(
        alignment.cigar)
    rpos, qpos = 0, alignment.q_st  # if alignment.strand == 1 else len(query) - alignment.q_en

    ref_to_query = np.empty((ref_len + 1, ), dtype=int)
    for length, op in cigar:
        if op == 0 or op == 7 or op == 8:  # Match or mismatch
            rend = rpos + length
            ref_to_query[rpos:rend] = qpos + np.arange(length)

            rpos = rend
            qpos += length
        elif op == 1:  # Insertion
            qpos += length
        elif op == 2:
            rend = rpos + length
            ref_to_query[rpos:rend] = qpos

            rpos = rend
        else:
            raise ValueError(
                f'Invalid cigar operation {op} for read {read_id}.')
    ref_to_query[rpos] = qpos  # Add the last one (excluded end)

    if ref_len != rpos:
        print(
            f'Warning: Mismatch between reference ({ref_len}) and cigar ({rpos}) length in query: {read_id}',
            file=sys.stderr)

    data = AlignmentData(alignment.ctg, alignment.r_st, alignment.r_en,
                         alignment.strand == 1, ref_to_query)
    return AlignmentInfo.SUCCESS, data


def get_aligner(reference_path: Path) -> mappy.Aligner:
    return mappy.Aligner(str(reference_path), preset='map-ont', best_n=1)
