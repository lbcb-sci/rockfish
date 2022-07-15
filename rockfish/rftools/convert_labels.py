from tqdm import tqdm, trange
import numpy as np

from io import BufferedReader

import argparse

from rockfish.rf_format import *


def logit_to_prob(logit: float) -> float:
    odds = np.exp(logit)
    return odds / (1 + odds)


def main(args: argparse.Namespace):
    tqdm.write('Started reading labels')
    labels = DictLabels(args.labels)

    if args.logits:
        tqdm.write('Logits are passed as input.')

    tqdm.write('Started parsing data')
    with open(args.data, 'rb') as rf_src, open(args.output, 'wb') as output:
        header = RFHeader.parse_header(rf_src)

        for i in trange(header.n_examples):
            example_header = RFExampleHeader.parse_bytes(
                rf_src.read(EXAMPLE_HEADER_STRUCT.size))
            rf_src.seek(example_header.example_len(args.seq_len), 1)

            ctg = header.ctgs[example_header.ctg_id]

            value = labels.get_label(example_header.read_id, ctg,
                                     example_header.pos)
            prob = logit_to_prob(value) if args.logits else value

            output.write(np.half(prob).tobytes())


def add_convert_labels_args(
        parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-l', '--labels', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-s', '--seq_len', type=int, default=31)
    parser.add_argument('--logits', action='store_true')