import argparse
import sys
from pathlib import Path
from typing import *

import gdown

MODELS = {
    'base': '1ajWkA361YrmXbnSbYhlzpJeqq2pG2J8P',
    'small': '1beWcPQJpO9W85bfFY438ophmzOxIrPdz'
}


def download_model(model: str, save_path: Path) -> None:
    try:
        id = MODELS[model]
    except KeyError:
        print(f"Model with name {model} doesn't exist.", file=sys.stderr)
        return

    path = save_path / f'rf_{model}.ckpt'
    gdown.download(id=id, output=str(path), quiet=False)


def download(args: argparse.Namespace) -> None:
    if args.save_path is None:
        save_path = Path(__file__).parent.resolve()
    else:
        save_path = args.save_path

    models = MODELS.keys() if 'all' in args.models else args.models
    for model in models:
        download_model(model, save_path)


def add_download_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-m', '--models', nargs='+', type=str, default='all')
    parser.add_argument('-s', '--save_path', type=Path, default=None)
