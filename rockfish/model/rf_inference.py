import argparse
import fileinput
import multiprocessing as mp
import os
import warnings
from contextlib import ExitStack
from typing import *

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import *
from .model import Rockfish

#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_flash_sdp(False)
#torch.backends.cuda.enable_math_sdp(True)


def parse_gpus(string: str) -> List[int]:
    if string is None:
        return None

    gpus = string.strip().split(',')
    return [int(g) for g in gpus]


def inference_worker(args: argparse.Namespace, gpu: Optional[int],
                     worker_out_path: str, start_idx: int, end_idx: int,
                     out_queue: mp.Queue) -> None:
    device = torch.device(f'cuda:{gpu}') if gpu is not None else torch.device(
        'cpu')

    with warnings.catch_warnings():
        model = Rockfish.load_from_checkpoint(args.ckpt_path,
                                              strict=False,
                                              track_metrics=False).to(device)
    model.eval()

    ref_len = model.hparams.bases_len
    block_size = model.block_size

    data = RFInferenceDataset(args.data_path, args.batch_size, ref_len,
                              block_size, start_idx, end_idx)
    loader = DataLoader(data,
                        args.batch_size,
                        shuffle=False,
                        num_workers=args.workers,
                        collate_fn=collate_fn_inference,
                        worker_init_fn=worker_init_rf_inference_fn,
                        pin_memory=True)
    with ExitStack() as manager:
        output_file = manager.enter_context(open(worker_out_path, 'w'))
        manager.enter_context(torch.no_grad())
        if gpu is not None:
            manager.enter_context(torch.cuda.amp.autocast())

        for ids, positions, signals, bases, n_blocks in loader:
            signals = signals.to(device)
            bases = bases.to(device)
            n_blocks = n_blocks.to(device)

            logits = model(signals, bases, n_blocks).cpu().numpy()

            for id, pos, logit in zip(ids, positions, logits):
                print(id, pos, logit, file=output_file, sep='\t')
            out_queue.put(len(positions))  # Notify tqdm


def cat_outputs(src_paths: List[str], dest_path: str) -> None:
    with open(dest_path, 'w') as dest:
        with fileinput.input(files=src_paths) as src:
            for line in src:
                dest.write(line)


def main(args: argparse.Namespace) -> None:
    gpus = parse_gpus(args.gpus) if args.gpus is not None else None

    n_processes = 1 if gpus is None else len(gpus)
    n_examples = get_n_examples(f'{args.data_path}.idx')
    per_process = int(math.ceil(n_examples / float(n_processes)))

    tqdm_queue = mp.Queue()
    output_paths = []
    for i in range(n_processes):
        gpu_id = gpus[i] if gpus is not None else None
        worker_out_path = args.output + f'.{i}.tmp'
        output_paths.append(worker_out_path)

        start_idx = i * per_process
        end_idx = min(start_idx + per_process, n_examples)

        process = mp.Process(
            target=inference_worker,
            args=(args, gpu_id, worker_out_path, start_idx, end_idx,
                  tqdm_queue),
            daemon=False)  # Daemonic processes cannot have children
        process.start()

    with tqdm(total=n_examples) as pbar:
        while pbar.n < n_examples:  # Processes should finish when this is false
            n_processed = tqdm_queue.get()
            pbar.update(n_processed)

    cat_outputs(output_paths, args.output)

    # Delete temporary files
    for path in output_paths:
        os.remove(path)


def add_rf_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('data_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('-o', '--output', type=str, default='predictions.tsv')
    parser.add_argument('-d', '--gpus', default=None)
    parser.add_argument('-t', '--workers', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
