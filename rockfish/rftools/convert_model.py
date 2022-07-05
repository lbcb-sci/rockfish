import torch

import argparse

from typing import *

INCLUDE_KEYS = {'hparams_name', 'hyper_parameters', 'state_dict'}
EXCLUDE_WEIGHTS = {'codebook.weight', 'fc_mask.weight', 'fc_mask.bias'}


def convert(args: argparse.Namespace) -> None:
    data = torch.load(args.src)
    result = {k: v for k, v in data.items() if k in INCLUDE_KEYS}

    state_dict = result['state_dict']
    for exclude_w in EXCLUDE_WEIGHTS:
        try:
            del state_dict[exclude_w]
        except KeyError:
            pass

    torch.save(result, args.dest)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', type=str, required=True)
    parser.add_argument('-d', '--dest', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    convert(args)
