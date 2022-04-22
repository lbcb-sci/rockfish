import sys
import argparse

from typing import *

from .extract.main import extract as extract_func
from .extract.main import add_extract_arguments as extract_args

from .rftools.index import main as index_func
from .rftools.index import add_index_arguments as index_args

from .rftools.merge import main as merge_func
from .rftools.merge import add_merge_arguments as merge_args

from .rftools.convert_labels import main as convert_labels_func
from .rftools.convert_labels import add_convert_labels_args as convert_labels_args


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    parser.set_defaults(func=lambda x: parser.print_usage())

    extract_parser = subparsers.add_parser('extract')
    extract_parser.set_defaults(func=extract_func)
    extract_args(extract_parser)

    index_parser = subparsers.add_parser('index')
    index_parser.set_defaults(func=index_func)
    index_args(index_parser)

    merge_parser = subparsers.add_parser('merge')
    merge_parser.set_defaults(func=merge_func)
    merge_args(merge_parser)

    convert_labels_parser = subparsers.add_parser('convert_labels')
    convert_labels_parser.set_defaults(func=convert_labels_func)
    convert_labels_args(convert_labels_parser)

    return parser.parse_args()


def main():
    args = get_arguments()
    args.func(args)


if __name__ == '__main__':
    main()
