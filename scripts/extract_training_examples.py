import os
os.environ['POLARS_MAX_THREADS'] = str(min(os.cpu_count(), 16))

import polars as pl
from tqdm import tqdm, trange

from pathlib import Path
import sys
import argparse

from rockfish.rf_format import *

from typing import *


LABEL_GROUP = pl.when(pl.col('label') > 0.5).then(1).otherwise(0).cast(pl.UInt8)


def parse_bedgraph(path: Path,
                   min_cov: int, 
                   include: Optional[List[str]]=None, 
                   exclude: Optional[List[str]]=None) -> pl.LazyFrame:
    df = pl.scan_csv(path, has_header=False, separator='\t', 
                     new_columns=['ctg', 'start', 'end', 'pct', 'n_mod', 'n_nomod'], 
                     dtypes={'ctg': pl.Categorical, 'start': pl.UInt32, 'n_mod': pl.UInt32, 'n_nomod': pl.UInt32})

    df = df.filter(((pl.col('n_mod') + pl.col('n_nomod')) >= min_cov) & 
                   ((pl.col('n_mod') == 0) | (pl.col('n_nomod') == 0)))

    if include is not None:
        df = df.filter(pl.col('ctg').is_in(include))
    elif exclude is not None:
        df = df.filter(pl.col('ctg').is_in(exclude).not_())

    df = df.select(['ctg', pl.col('start').alias('ref_pos'), 
                    pl.when(pl.col('pct') > 50).then(1.0).otherwise(0.0).cast(pl.Float32).alias('label')])

    return df

def parse_mappings(path: Path) -> pl.LazyFrame:
    df = pl.scan_csv(path, has_header=False, separator='\t',
                     new_columns=['read_id', 'read_pos', 'ref_pos', 'ctg'],
                     dtypes={'read_id': pl.Utf8, 'read_pos': pl.UInt32, 'ref_pos': pl.UInt32, 'ctg': pl.Categorical})
    
    return df


def rf_extract_info(path: Path) -> pl.LazyFrame:
    with path.open('rb') as f:
        header = RFHeader.parse_header(f)

        examples = []
        for _ in trange(10_000_000):
            fpos = f.tell()
            example = RFExample.from_file(f, 31)

            read_id = example.header.read_id
            read_pos = example.header.pos

            examples.append((read_id, read_pos, fpos))

    df = pl.LazyFrame(examples, 
                      schema={'read_id': pl.Utf8, 'read_pos': pl.UInt32, 'fpos': pl.UInt64})
    return df


def main(args: argparse.Namespace):
    pl.enable_string_cache()

    bedgraph = parse_bedgraph(args.bedgraph, args.min_cov, args.include_ctgs, args.exclude_ctgs)
    mappings = parse_mappings(args.mappings)
    rf_data = rf_extract_info(args.rf)

    df = bedgraph.join(mappings, on=['ctg', 'ref_pos'], how='inner') \
                 .join(rf_data, on=['read_id', 'read_pos'], how='inner')

    if args.balanced:
        counts = df.group_by(LABEL_GROUP) \
                .agg(pl.count()) \
                .collect()
        
        n_per_class = counts['count'].min()
        if args.n_per_class is not None:
            if args.n_per_class > n_per_class:
                tqdm.write(f'Warning: Provided number of examples per class is higher than possible.')
            else:    
                n_per_class = args.n_per_class

        tqdm.write(f'Sampling {n_per_class} examples per class.')

        df = df.filter(
            pl.int_range(0, pl.count()).shuffle(seed=args.seed).over(LABEL_GROUP) < n_per_class
        )
        
    for row in df.select(['fpos', 'label']).collect().iter_rows():
        print(row)

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--rf', type=Path, required=True)
    parser.add_argument('--mappings', type=Path, required=True)
    parser.add_argument('--bedgraph', type=Path, required=True)

    parser.add_argument('--min_cov', type=int, default=30)

    ctgs = parser.add_mutually_exclusive_group()
    ctgs.add_argument('--include_ctgs', nargs='+')
    ctgs.add_argument('--exclude_ctgs', nargs='+')

    parser.add_argument('--balanced', action='store_true', 
                        required='--n_per_class' in ''.join(sys.argv))
    parser.add_argument('--n_per_class', type=int)

    parser.add_argument('--seed', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)