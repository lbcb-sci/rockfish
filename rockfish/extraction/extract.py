from ctypes import alignment
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api.fast5_read import Fast5Read
from tqdm import tqdm
import mappy

from pathlib import Path
import re
import argparse

from typing import *

from fast5 import get_read_info
from alignment import get_aligner, align_read


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


MotifPositions = dict[str, Tuple[Set[int], Set[int]]]
def build_reference_idx(aligner: mappy.Aligner, motif: str, 
                        rel_idx: int) -> MotifPositions:
    positions = OrderedDict()

    for contig in aligner.seq_names:
        sequence = aligner.seq(contig)

        fwd_pos = {m.start() + rel_idx for m in re.finditer(motif, sequence, re.I)}

        rev_comp = mappy.revcomp(sequence)
        def pos_for_rev(i: int) -> int:
            return len(sequence) - (i + rel_idx) - 1
        rev_pos = {pos_for_rev(m.start()) 
                   for m in re.finditer(motif, rev_comp, re.I)}

        positions[contig] = (fwd_pos, rev_pos)

    return positions


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('dest', type=str)

    parser.add_argument('--motif', type=str, default='CG')
    parser.add_argument('--idx', type=int, default=0)

    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('--window', type=int, default=15)

    parser.add_argument('--n_readers', type=int, default=1)
    parser.add_argument('--n_processors', type=int, default=1)
    parser.add_argument('--n_writers', type=int, default=1)

    parser.add_argument('--info_file', type=str, default='info.txt')

    parser.add_argument('--read_ids', type=Path, default=None)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference)
    ref_positions = build_reference_idx(aligner, args.motif, args.idx)

    tqdm.write(f'Retrieving files from {args.source}, recursive {args.recursive}')
    files = get_files(args.source, args.recursive)

    count = 0
    for f in files:
        for r in get_reads(f):
            read_info = get_read_info(r)
            aln_info = align_read(read_info.sequence, aligner)
            if aln_info is not None:
                idx = aln_info.ref_to_query[0]
                print(idx, len(read_info.seq_to_sig), aln_info.fwd_strand)
                break
        

    print(count)


if __name__ == '__main__':
    args = get_arguments()

    main(args)
