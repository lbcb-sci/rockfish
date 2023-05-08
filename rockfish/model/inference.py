import argparse
import random
import sys
import traceback
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import *

import mappy
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, IterableDataset

from rockfish.extract.extract import Example, build_reference_idx2
from rockfish.extract.main import *
from rockfish.model.datasets import *
from rockfish.model.model import Rockfish

MIN_BLOCKS_LEN_FACTOR = 1
MAX_BLOCKS_LEN_FACTOR = 4

HEADER = '\t'.join(['read_id', 'ctg', 'pos', 'prob'])


def parse_gpus(string: str) -> List[int]:
    if string is None:
        return None

    gpus = string.strip().split(',')
    return [int(g) for g in gpus]


def load_model(path: str, device: str, gpus: List[int]):
    with warnings.catch_warnings():
        model = Rockfish.load_from_checkpoint(path,
                                              strict=False,
                                              track_metrics=False)

    block_size = model.block_size

    if gpus is not None and len(gpus) > 1:
        model = DataParallel(model, gpus)
    model = model.to(device)

    return model, block_size


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
                 unique_aln: bool, batch_size: int, block_size: int,
                 device: str) -> None:
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
        self.device = device

        self.mapping_encodings = ReferenceMapping(self.bases_len, block_size)

    def __iter__(self):
        '''bins = ExampleBins(self.bases_len * MIN_BLOCKS_LEN_FACTOR,
                           self.bases_len * MAX_BLOCKS_LEN_FACTOR,
                           self.batch_size)'''
        bins = ExampleBins(16, 155, self.batch_size)

        model_kmers_path = files('rockfish.extract').joinpath('model_kmers.txt')
        model_kmers = load_model_kmers(model_kmers_path)

        buffer = mappy.ThreadBuffer()
        for file in self.files:
            for read in get_reads(file):
                try:
                    read_info = load_read(read)
                except Exception as e:
                    print(f'Cannot load read from file {file}.',
                          file=sys.stderr)
                    # traceback.print_exc()
                    continue

                try:
                    _, examples = extract_features(read_info,
                                                   self.ref_positions,
                                                   self.aligner, buffer,
                                                   model_kmers, self.window,
                                                   self.mapq_filter,
                                                   self.unique_aln)
                except Exception as e:
                    print(
                        f'Cannot process read {read_info.read_id} from file {file}.',
                        file=sys.stderr)
                    # traceback.print_exc()
                    continue

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
        signal = torch.tensor(
            example.signal,
            dtype=torch.float if self.device == 'cpu' else torch.half).unfold(
                -1, self.block_size, self.block_size)
        bases = torch.tensor([ENCODING.get(b, 4) for b in example.bases])
        mean_diffs = torch.tensor((example.diff_means - DIFF_MEAN) / DIFF_STD, dtype=torch.float)
        q_indices = torch.tensor(example.q_indices.astype(np.int32))
        lengths = torch.tensor(np.array(example.event_length).astype(np.int32))

        r_pos_enc = self.mapping_encodings(lengths)

        return example.read_id, example.ctg, example.pos, signal, bases, mean_diffs, r_pos_enc, q_indices


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    total_files = len(dataset.files)
    per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))

    start = worker_id * per_worker
    end = min(start + per_worker, total_files)
    dataset.files = dataset.files[start:end]


def inference(args: argparse.Namespace) -> None:
    files = list(get_files(args.input, args.recursive))
    random.shuffle(files)

    tqdm.write(f'Parsing reference file {args.reference}')
    aligner = get_aligner(args.reference, args.workers)

    tqdm.write('Building reference positions for the given motif.')
    ref_positions = build_reference_idx2(aligner, args.motif, args.idx,
                                         args.workers)

    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    device = 'cpu' if gpus is None else gpus[0]

    model, block_size = load_model(args.model_path, device, gpus)
    model.eval()

    dataset = Fast5Dataset(files, ref_positions, aligner, args.window,
                           args.mapq_filter, args.unique_aln, args.batch_size,
                           block_size, device)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        collate_fn=collate_fn_inference,
                        worker_init_fn=worker_init_fn,
                        pin_memory=True)

    with ExitStack() as manager:
        output_file = manager.enter_context(open(args.output, 'w'))
        output_file.write(f'{HEADER}\n')

        manager.enter_context(torch.no_grad())
        pbar = manager.enter_context(tqdm())

        if gpus is not None:
            manager.enter_context(torch.cuda.amp.autocast())

        for ids, ctgs, positions, signals, bases, mean_diffs, r_mappings, q_mappings, n_blocks in loader:
            signals = signals.to(device)
            bases = bases.to(device)
            mean_diffs = mean_diffs.to(device)
            r_mappings = r_mappings.to(device)
            q_mappings = q_mappings.to(device)
            n_blocks = n_blocks.to(device)

            probs = model(signals, r_mappings, q_mappings, bases, mean_diffs,
                          n_blocks).sigmoid().cpu().numpy()

            for id, ctg, pos, prob in zip(ids, ctgs, positions, probs):
                print(id, ctg, pos, prob, file=output_file, sep='\t')

            pbar.update(n=len(positions))


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-i', '--input', type=Path, required=True)
    parser.add_argument('-r', '--recursive', action='store_true')

    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--motif', type=str, default='CG')
    parser.add_argument('--idx', type=int, default=0)

    parser.add_argument('-w', '--window', type=int, default=15)
    parser.add_argument('-q', '--mapq_filter', type=int, default=0)
    parser.add_argument('-u', '--unique_aln', action='store_true')

    parser.add_argument('-d', '--gpus', default=None)
    parser.add_argument('-t', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4096)
    # parser.add_argument('--combined_mask', action='store_true')

    parser.add_argument('-o', '--output', type=str, default='predictions.tsv')


if __name__ == '__main__':
    args = add_inference_arguments(argparse.ArgumentParser())
    inference(args)
