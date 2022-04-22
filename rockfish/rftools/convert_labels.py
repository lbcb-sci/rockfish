from tqdm import tqdm, trange
import numpy as np

from io import BufferedReader

import argparse

from rockfish.rf_format import *


def main(args: argparse.Namespace):
    labels = DictLabels(args.labels)

    with open(args.data) as rf_src:
        header = RFHeader.parse_header(rf_src)

        probs = np.empty((header.n_examples,), dtype=np.half)
        for i in trange(header.n_examples):
            example_header = RFExampleHeader.parse_bytes(rf_src.read(EXAMPLE_HEADER_STRUCT))
            rf_src.seek(example_header.example_len(args.seq_len), 1)

            ctg = header.ctgs[example_header.ctg_id]
            prob = labels.get_label(example_header.read_id, ctg, example_header.pos)
            probs[i] = prob

    np.save(args.output, probs)


def add_convert_labels_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-l', '--labels', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-s', '--seq_len', type=int, default=31)

    return parser.parse_args()
