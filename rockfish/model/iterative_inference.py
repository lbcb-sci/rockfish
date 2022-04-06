from contextlib import ExitStack
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import multiprocessing as mp
import argparse
import fileinput

from datasets import *
from model import Rockfish

from typing import *


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

    model = Rockfish.load_from_checkpoint(args.ckpt_path,
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


@torch.no_grad()
def inference(args: argparse.Namespace) -> None:
    model = Rockfish.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.freeze()

    ref_len = model.hparams.bases_len
    block_size = model.block_size

    gpus = parse_gpus(args.gpus) if args.gpus is not None else None
    if gpus is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpus[0]}')
        if len(gpus) > 1:
            model = DataParallel(model, device_ids=gpus)
    else:
        device = torch.device('cpu')
    model.to(device)

    data = RFInferenceDataset(args.data_path,
                              args.batch_size,
                              ref_len=ref_len,
                              block_size=block_size)
    loader = DataLoader(data,
                        args.batch_size,
                        False,
                        num_workers=args.workers,
                        collate_fn=collate_fn_inference,
                        worker_init_fn=worker_init_rf_inference_fn,
                        pin_memory=True)

    with open(args.output, 'w') as f, tqdm(
            total=len(data.offsets)) as pbar, torch.cuda.amp.autocast():
        for ids, ctgs, poss, signals, bases, r_pos_enc, q_indices, num_blocks in loader:
            signals, bases, r_pos_enc, q_indices, num_blocks = (
                signals.to(device), bases.to(device), r_pos_enc.to(device),
                q_indices.to(device), num_blocks.to(device))

            logits = model(signals, bases, r_pos_enc, q_indices,
                           num_blocks).cpu().numpy()  # N

            for id, ctg, pos, logit in zip(ids, ctgs, poss, logits):
                print(id, ctg, pos, logit, file=f, sep='\t')
            pbar.update(len(logits))


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path', type=str)
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('-o', '--output', type=str, default='predictions.tsv')
    parser.add_argument('-d', '--gpus', default=None)
    parser.add_argument('-t', '--workers', type=int, default=0)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    # parser.add_argument('-s', '--ref_len', type=int, default=31) Infer from the model
    # parser.add_argument('--block_size', type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    # inference(args)
    main(args)
