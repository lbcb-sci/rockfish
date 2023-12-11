import argparse
from pathlib import Path

import pysam
import pod5 as p5
import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from pod5.reader import ReadRecord
from pysam import AlignedSegment

from .fast5 import ReadInfo

from typing import *

np.seterr(divide='raise')

@dataclass
class BamIndex:
    bampath: str

    def __post_init__(self):
        self.bam_f = None
        self.num_recs = 0
        self.aligned = False
        self.build_index()

    def open_bam(self):
        self.bam_f = pysam.AlignmentFile(self.bampath, 'rb', check_sq=False)

    def close_bam(self):
        self.bam_f.close()
        self.bam_f = None

    def build_index(self):
        if self.bam_f is None:
            self.open_bam()
        self.bam_idx = defaultdict(list)
        tqdm.write('Indexing BAM file by read ids')

        while True:
            read_ptr = self.bam_f.tell()
            try:
                read = next(self.bam_f)
            except StopIteration:
                tqdm.write('Finished reading bam file')
                break
            read_id = read.query_name
            if read.is_supplementary or read.is_secondary or read.read_id in self.bam_idx:
                continue
            self.num_recs += 1
            self.bam_idx[read_id].append(read_ptr)
        self.close_bam()
        self.bam_idx = dict(self.bam_idx)
        self.num_reads = len(self.bam_idx)

    def get_alignment(self, read_id: str) -> AlignedSegment:
        if self.bam_f is None:
            self.open_bam()
        try:
            read_ptrs = self.bam_idx[read_id]
        except KeyError:
            tqdm.write(f'Cannot find read {read_id} in bam index')
            return None
        for read_ptr in read_ptrs:
            self.bam_f.seek(read_ptr)
            try:
                bam_read = next(self.bam_f)
            except OSError:
                tqdm.write(f'Cannot extract read {read_id} from bam index')
                continue
            assert str(bam_read.query_name) == read_id, (tqdm.write(f'Given read id {read_id} does not match read retrieved '
                                                               f'from bam index {bam_read.query_name}'))
            yield bam_read


@dataclass
class PodReadInfo:
    read_id: str
    signal: np.ndarray
    scale: float
    offset: float

    def __post_init__(self):
        self.fastq = ''
        self.quals = None
        self.move_table = np.array([])
        self.block_stride = 0

    def update_from_bam(self, bam_data):
        self.fastq = bam_data.get_forward_sequence()
        self.quals = bam_data.get_forward_qualities()
        mv_data = bam_data.get_tag('mv')
        self.block_stride = mv_data.pop(0)
        self.move_table = np.array(mv_data)
        raw_start = bam_data.get_tag('ts')
        self.signal = self.signal[raw_start:]
        self.signal = self.calibrate()

    def calibrate(self):
        return np.array(self.scale * (self.signal + self.offset), dtype=np.float32)

    def get_seq_and_quals(self) -> Tuple[str, np.ndarray]:
        return self.fastq, self.quals

    def get_seq_to_sig(self) -> np.ndarray:
        move_table = np.append(self.move_table,
                               1)  # Adding for easier indexing
        return move_table.nonzero()[0] * self.block_stride

    def get_normalized_signal(self, start=0, end=None) -> np.ndarray:
        signal = self.signal[start:end]
        med = np.median(signal)
        mad = np.median(np.abs(signal - med))

        return (signal - med) / (1.4826 * mad)


def get_readid_overlaps(bam_idx: BamIndex, pod5_read_ids: List[str]) -> List[str]:
    overlap_read_ids = list(set(pod5_read_ids).intersection(bam_idx.bam_idx.keys()))
    return overlap_read_ids


def load_pod5_read(read: ReadRecord) -> PodReadInfo:
    return PodReadInfo(read_id=str(read.read_id), signal=read.signal, scale=read.calibration.scale, offset=read.calibration.offset)


def load_signals(pod5_file: Path, read_ids: List[str]) -> Generator[ReadRecord, None, None]:
    with p5.Reader(path=pod5_file) as reader:
        yield from reader.reads(selection=read_ids)


def match_pod5_and_bam(args: argparse.Namespace, files: List[Path]):
    bam_idx = BamIndex(bampath=args.bam_path)
    tqdm.write(f'Bam indexed with {bam_idx.num_reads} reads')
    pod5_file_readid_pairs = []
    for f in files:
        with p5.Reader(f) as pod5_f:
            read_ids = get_readid_overlaps(bam_idx, pod5_f.read_ids)
            #tqdm.write(f'Extracted {len(read_ids)} overlapping reads from {f} and BAM index')
            pod5_file_readid_pairs.append((f, read_ids))

    return bam_idx, pod5_file_readid_pairs



