from __future__ import annotations

from ont_fast5_api.fast5_read import Fast5Read

from dataclasses import dataclass
import numpy as np

from typing import *

FASTQ_PATH = 'BaseCalled_template/Fastq'
MOVE_TABLE_PATH = 'BaseCalled_template/Move'


@dataclass
class ReadInfo:
    read_id: str
    fastq: str
    signal: np.ndarray
    move_table: np.ndarray
    block_stride: int

    def get_seq_to_sig(self) -> np.ndarray:
        move_table = np.append(self.move_table, 1)  # Adding for easier indexing
        return move_table.nonzero()[0] * self.block_stride

    def get_seq_and_quals(self) -> Tuple[str, np.np.ndarray]:
        data = self.fastq.strip().split('\n')

        sequence = data[1]
        quals = np.array([ord(c) - 33 for c in data[3]], dtype=np.uint8)

        return sequence, quals

    def get_normalized_signal(self, start=0, end=None) -> np.ndarray:
        signal = self.signal[start:end]
        med = np.median(signal)
        mad = np.median(np.abs(signal - med))

        return (signal - med) / (1.4826 * mad)


def load_read(read: Fast5Read) -> ReadInfo:
    bc_analysis = read.get_latest_analysis('Basecall_1D')
    bc_summary = read.get_summary_data(bc_analysis)

    block_stride = bc_summary['basecall_1d_template']['block_stride']
    move_table = read.get_analysis_dataset(bc_analysis, MOVE_TABLE_PATH)

    fastq = read.get_analysis_dataset(bc_analysis, FASTQ_PATH)
    if fastq is None:
        raise ValueError('Fastq data is empty.')

    seg_analysis = read.get_latest_analysis('Segmentation')
    seg_summary = read.get_summary_data(seg_analysis)
    start = seg_summary['segmentation']['first_sample_template']

    signal = read.get_raw_data(start=start, scale=True)
    if len(signal) == 0:
        raise ValueError('Signal array is empty.')

    return ReadInfo(read.read_id, fastq, signal, move_table, block_stride)
