import torch
from torch.nn import DataParallel
from torch.utils.data import IterableDataset, DataLoader
import mappy

import random
from pathlib import Path
from contextlib import ExitStack
import sys
import traceback
import argparse

from rockfish.extract.extract import Example
from rockfish.extract.main import *
from rockfish.model.model import Rockfish
from rockfish.model.datasets import *

from typing import *

MIN_BLOCKS_LEN_FACTOR = 1
MAX_BLOCKS_LEN_FACTOR = 4


def parse_gpus(string: str) -> List[int]:
    if string is None:
        return None

    gpus = string.strip().split(',')
    return [int(g) for g in gpus]


def example_to_tensor(
    self, example: Example
) -> Tuple[str, str, int, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    signal = torch.tensor(example.signal,
                          dtype=torch.half).unfold(-1, self.block_size,
                                                   self.block_size)
    bases = torch.tensor([ENCODING.get(b, 4) for b in example.data.bases])
    q_indices = torch.tensor(example.data.q_indices.astype(np.int32))
    lengths = torch.tensor(example.data.event_lengths.astype(np.int32))

    r_pos_enc = self.mapping_encodings(lengths)

    return example.header.read_id, self.ctgs[
        example.header.
        ctg_id], example.header.pos, signal, bases, r_pos_enc, q_indices


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

            return bin

        if self.count >= self.max_examples:
            return self.emit_batch()

    def emit_batch(self) -> Iterator[Example]:
        batch_processed = 0

        for bin in reversed(self.bins):
            while len(bin) > 0 and batch_processed < self.batch_size:
                example = bin.pop()

                batch_processed += 1
                self.count -= 1

                yield example

    def emit_all(self) -> Iterator[Example]:
        for bin in self.bins:
            for example in bin:
                yield example


class Fast5Dataset(IterableDataset):

    def __init__(self, files: List[Path], ref_positions: MotifPositions,
                 aligner: mappy.Aligner, window: int, mapq_filter: int,
                 unique_aln: bool, batch_size: int, block_size: int) -> None:
        super().__init__()

        self.files = files
        self.ref_positions = ref_positions
        self.aligner = aligner
        self.window = window
        self.mapq_filter = mapq_filter
        self.unique_aln = unique_aln
        self.block_size = block_size
        self.bases_len = (2 * window) + 1
        self.batch_size = batch_size

        self.mapping_encodings = ReferenceMapping(self.bases_len, block_size)

    def __iter__(self):
        bins = ExampleBins(self.bases_len * MIN_BLOCKS_LEN_FACTOR,
                           self.bases_len * MAX_BLOCKS_LEN_FACTOR,
                           self.batch_size)

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

                if examples is None:
                    continue

                for example in examples:
                    batch = bins.add_example(example)
                    if batch is not None:
                        yield from (self.example_to_tensor(e) for e in batch)

        for example in bins.emit_all():
            yield self.example_to_tensor(example)

    def example_to_tensor(
        self, example: Example
    ) -> Tuple[str, str, int, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        signal = torch.tensor(example.signal,
                              dtype=torch.half).unfold(-1, self.block_size,
                                                       self.block_size)
        bases = torch.tensor([ENCODING.get(b, 4) for b in example.bases])
        q_indices = torch.tensor(example.q_indices.astype(np.int32))
        lengths = torch.tensor(np.array(example.event_length).astype(np.int32))

        r_pos_enc = self.mapping_encodings(lengths)

        return example.read_id, example.ctg, example.pos, signal, bases, r_pos_enc, q_indices


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    total_files = len(dataset.files)
    per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))

    start = worker_id * per_worker
    end = min(start + per_worker, total_files)
    dataset.files = dataset.files[start:end]


def main(args: argparse.Namespace) -> None:
    files = list(get_files(args.input, args.recursive))
    random.shuffle(files)

    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference)

    tqdm.write('Building reference positions for the given motif.')
    ref_positions = build_reference_idx(aligner, args.motif, args.idx)

    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    device = 'cpu' if gpus is None else gpus[0]

    model = Rockfish.load_from_checkpoint(args.model_path, track_metrics=False)
    if gpus is None:
        model = model.to('cpu')
    if len(gpus) == 1:
        model = model.to(device)
    else:
        model = DataParallel(model, gpus)
    model.eval()

    dataset = Fast5Dataset(files, ref_positions, aligner, args.window,
                           args.mapq_filter, args.unique_aln, args.batch_size,
                           model.block_size)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        collate_fn=collate_fn_inference,
                        worker_init_fn=worker_init_fn,
                        pin_memory=True)

    with ExitStack() as manager:
        output_file = manager.enter_context(open(args.output, 'w'))
        manager.enter_context(torch.no_grad())
        pbar = manager.enter_context(tqdm())

        if gpus is not None:
            manager.enter_context(torch.cuda.amp.autocast())

        for ids, ctgs, positions, signals, bases, r_mappings, q_mappings, n_blocks in loader:
            signals = signals.to(device)
            bases = bases.to(device)
            r_mappings = r_mappings.to(device)
            q_mappings = q_mappings.to(device)
            n_blocks = n_blocks.to(device)

            logits = model(signals, r_mappings, q_mappings, bases,
                           n_blocks).cpu().numpy()

            for id, ctg, pos, logit in zip(ids, ctgs, positions, logits):
                print(id, ctg, pos, logit, file=output_file, sep='\t')

            pbar.update(n=len(positions))


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=Path, required=True)
    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--reference', type=str)
    parser.add_argument('--motif', type=str, default='CG')
    parser.add_argument('--idx', type=int, default=0)

    parser.add_argument('-w', '--window', type=int, default=15)
    parser.add_argument('-q', '--mapq_filter', type=int, default=0)
    parser.add_argument('-u', '--unique_aln', action='store_true')

    parser.add_argument('-d', '--gpus', default=None)
    parser.add_argument('-t', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4096)

    parser.add_argument('-o', '--output', type=str, default='predictions.tsv')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)
