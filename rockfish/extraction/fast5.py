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
    signal: np.ndarray
    sequence: str
    quals: np.ndarray
    seq_to_sig: np.ndarray


def get_signal(read: Fast5Read, length: int) -> np.ndarray:
    seg_analysis = read.get_latest_analysis('Segmentation')
    seg_summary = read.get_summary_data(seg_analysis)
    start = seg_summary['segmentation']['first_sample_template']

    return read.get_raw_data(start=start, end=start+length, scale=True)


def map_sequence_to_signal(read: Fast5Read) -> np.ndarray:
    bc_analysis = read.get_latest_analysis('Basecall_1D')
    bc_summary = read.get_summary_data(bc_analysis)

    block_stride = bc_summary['basecall_1d_template']['block_stride']

    move_table = read.get_analysis_dataset(bc_analysis, MOVE_TABLE_PATH)
    move_table = np.append(move_table, 1)  # Adding for easier indexing

    seq_to_sig = move_table.nonzero()[0] * block_stride
    return seq_to_sig, block_stride


def get_fastq_data(read: Fast5Read) -> Tuple[str, np.ndarray]:
    latest_basecall = read.get_latest_analysis('Basecall_1D')

    dataset = read.get_analysis_dataset(latest_basecall, FASTQ_PATH)
    data = dataset.strip().split('\n')

    sequence = data[1]
    quals = np.array([ord(c) - 33 for c in data[3]], dtype=np.uint8)

    return sequence, quals


def get_read_info(read: Fast5Read) -> ReadInfo:
    seq_to_sig, _ = map_sequence_to_signal(read)
    signal = get_signal(read, seq_to_sig[-1])
    sequence, qualities = get_fastq_data(read)

    return ReadInfo(read.read_id, signal, sequence, qualities, seq_to_sig)
