from torch.utils.data import IterableDataset
import mappy

from pathlib import Path
import sys
import traceback

from rockfish.extract.extract import Example
from rockfish.extract.main import *

from typing import *


class ExampleBins:

    def __init__(self,
                 min_len: int,
                 max_len: int,
                 batch_size: int,
                 storage_factor: int = 4,
                 bin_range: int = 10) -> None:
        self.offset = min_len // bin_range
        n_bins = (max_len // bin_range) - self.offset + 1

        self.bins = [[] for _ in range(n_bins)]
        self.count = 0

        self.batch_size = batch_size
        self.max_examples = batch_size * storage_factor

    def add_example(self, example: Example) -> Iterator[Example]:
        n_bin = len(example.q_indices) // 10 - self.offset
        bin = self.bins[n_bin]

        bin.append(example)
        self.count += 1

        if len(bin) >= self.batch_size:
            self.bins[n_bin] = []
            self.count -= len(bin)

            return self.bins

        if self.count >= self.max_examples:
            return self.emit_batch()

    def emit_batch(self) -> Iterator[Example]:
        batch_processed = 0

        for bin in self.bins:
            while len(bin) > 0 and batch_processed < self.batch_size:
                example = bin.pop()

                batch_processed += 1
                self.count -= 1

                yield example


class Fast5Dataset(IterableDataset):

    def __init__(self, files: List[Path], ref_positions: MotifPositions,
                 aligner: mappy.Aligner, window: int, mapq_filter: int,
                 unique_aln: bool) -> None:
        super().__init__()

        self.files = files
        self.ref_positions = ref_positions
        self.aligner = aligner
        self.window = window
        self.mapq_filter = mapq_filter
        self.unique_aln = unique_aln

    def __iter__(self):
        self.bins = ExampleBins()

        buffer = mappy.ThreadBuffer()
        for file in self.files:
            for read in get_reads(file):
                try:
                    read_info = load_read(read)
                except Exception as e:
                    print(f'Cannot load read from file {file}.',
                          file=sys.stderr)
                    traceback.print_exc()
                    sys.exit(-1)

                try:
                    read_info = load_read(read)
                    _, examples = extract_features(
                        read_info, self.ref_positions, self.aligner, buffer,
                        self.window, self.mapq_filter, self.unique_aln)
                except Exception as e:
                    print(
                        f'Cannot process read {read_info.read_id} from file {file}.',
                        file=sys.stderr)
                    traceback.print_exc()
                    sys.exit(-1)
