import numpy as np
import mappy

import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
import re

from typing import *

from .fast5 import ReadInfo
from .pod5 import  BamIndex, PodReadInfo
from .alignment import AlignmentData, AlignmentInfo, align_read

MotifPositions = Dict[str, Tuple[Set[int], Set[int]]]

MIN_BLOCKS_LEN_FACTOR = 0.5
MAX_BLOCKS_LEN_FACTOR = 5

@dataclass
class Example:
    read_id: str
    ctg: str
    pos: int
    signal: np.ndarray
    event_length: List[int]
    bases: str
    q_indices: np.ndarray


def build_reference_idx(aligner: mappy.Aligner, motif: str,
                        rel_idx: int) -> MotifPositions:
    positions = OrderedDict()

    for contig in aligner.seq_names:
        sequence = aligner.seq(contig)

        fwd_pos = {
            m.start() + rel_idx for m in re.finditer(motif, sequence, re.I)
        }

        rev_comp = mappy.revcomp(sequence)
        seq_len = len(rev_comp)

        def pos_for_rev(i: int) -> int:
            return seq_len - (i + rel_idx) - 1

        rev_pos = {
            pos_for_rev(m.start()) for m in re.finditer(motif, rev_comp, re.I)
        }

        positions[contig] = (fwd_pos, rev_pos)

    return positions


def build_index_for_ctg(sequence: str, motif: str,
                        rel_idx: str) -> Tuple[Set[int], Set[int]]:
    fwd_pos = {m.start() + rel_idx for m in re.finditer(motif, sequence, re.I)}

    rev_comp = mappy.revcomp(sequence)
    seq_len = len(rev_comp)

    def pos_for_rev(i: int) -> int:
        return seq_len - (i + rel_idx) - 1

    rev_pos = {
        pos_for_rev(m.start()) for m in re.finditer(motif, rev_comp, re.I)
    }

    return fwd_pos, rev_pos


def build_reference_idx2(aligner: mappy.Aligner, motif: str, rel_idx: int,
                         n_workers: int) -> MotifPositions:
    ctgs = [ctg for ctg in aligner.seq_names]

    n_workers = min(n_workers, len(ctgs))
    with mp.Pool(n_workers) as pool:
        idx_func = partial(build_index_for_ctg, motif=motif, rel_idx=rel_idx)
        positions = pool.map(idx_func, (aligner.seq(ctg) for ctg in ctgs))

    return {c: p for c, p in zip(ctgs, positions)}


def get_ref_pos(aln_data: AlignmentData, ref_positions: MotifPositions,
                window: int) -> Iterator[int]:
    if aln_data.fwd_strand:
        ctg_pos = ref_positions[aln_data.ctg][0]
    else:
        ctg_pos = ref_positions[aln_data.ctg][1]

    if aln_data.fwd_strand:
        rng = range(aln_data.r_start + window, aln_data.r_end - window)
    else:
        rng = range(aln_data.r_end - 1 - window, aln_data.r_start - 1 + window,
                    -1)

    for rel, rpos in enumerate(rng, start=window):
        if rpos in ctg_pos:
            yield rel, rpos


def get_event_length(position: int, ref_to_query: np.ndarray,
                     query_to_signal: np.ndarray) -> int:
    q_st, q_en = ref_to_query[position], ref_to_query[position + 1]
    s_st, s_en = query_to_signal[q_st], query_to_signal[q_en]
    return s_en - s_st


def extract_features(read_info: ReadInfo, ref_positions: MotifPositions,
                     aligner: mappy.Aligner, buffer: mappy.ThreadBuffer,
                     window: int, mapq_filter: int,
                     unique_aln: bool) -> Tuple[AlignmentInfo, List[Example]]:
    seq_to_sig = read_info.get_seq_to_sig()
    signal = read_info.get_normalized_signal(end=seq_to_sig[-1]) \
                        .astype(np.half)
    query, _ = read_info.get_seq_and_quals()
    example_bases = (2 * window) + 1

    status, aln_data = align_read(query, aligner, buffer, mapq_filter,
                                  unique_aln, read_info.read_id)
    if aln_data is None:
        return status, None

    ref_seq = aligner.seq(aln_data.ctg, aln_data.r_start, aln_data.r_end)
    ref_seq = ref_seq if aln_data.fwd_strand else mappy.revcomp(ref_seq)

    examples = []
    for rel, pos in get_ref_pos(aln_data, ref_positions, window):
        q_start = aln_data.ref_to_query[rel - window]
        sig_start = seq_to_sig[q_start]

        # q_end -> Start of the first base after example
        q_end = aln_data.ref_to_query[rel + window + 1]
        # sig_end -> Start of the first signal point after example
        sig_end = seq_to_sig[q_end]

        n_blocks = (sig_end - sig_start) // read_info.block_stride
        if n_blocks < example_bases or n_blocks > 4 * example_bases:
            continue

        move_start = (sig_start - seq_to_sig[0]) // read_info.block_stride
        move_end = (sig_end - seq_to_sig[0]) // read_info.block_stride
        q_indices = read_info.move_table[move_start:move_end].cumsum() - 1

        event_lengts = [
            get_event_length(p, aln_data.ref_to_query, seq_to_sig)
            for p in range(rel - window, rel + window + 1)
        ]

        example = Example(read_info.read_id, aln_data.ctg, pos,
                          signal[sig_start:sig_end], event_lengts,
                          ref_seq[rel - window:rel + window + 1], q_indices)
        examples.append(example)

    return status, examples


def extract_pod5_features(read_info: PodReadInfo, bamidx: BamIndex, ref_positions: MotifPositions,
                     aligner: mappy.Aligner, buffer: mappy.ThreadBuffer,
                     window: int, mapq_filter: int,
                     unique_aln: bool) -> Tuple[AlignmentInfo, List[Example]]:

    for bam_data in bamidx.get_alignment(read_info.read_id):
        read_info.update_from_bam(bam_data)
        status, examples = extract_features(read_info=read_info, ref_positions=ref_positions, aligner=aligner,
                                        buffer=buffer, window=window, mapq_filter=mapq_filter,
                                        unique_aln=unique_aln)
        yield status, examples
