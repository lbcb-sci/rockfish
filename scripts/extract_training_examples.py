import os
os.environ['POLARS_MAX_THREADS'] = str(min(os.cpu_count(), 8))

import polars as pl
from tqdm import tqdm, trange

from pathlib import Path
import sys
import argparse

from rockfish.rf_format import *

from typing import *


LABEL_GROUP = pl.when(pl.col('label') > 0.5).then(1).otherwise(0).cast(pl.UInt8)
RF_EXTRACT_DTYPE = np.dtype([('read_id', 'U36'), ('read_pos', 'u4'), ('fpos', 'u8')])


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


def rf_extract_info(path: Path, seq_len: int=31) -> pl.LazyFrame:
    with path.open('rb') as f:
        header = RFHeader.parse_header(f)

        examples = np.empty(header.n_examples, RF_EXTRACT_DTYPE)
        for i in trange(header.n_examples):
            fpos = f.tell()
            example = RFExample.from_file(f, seq_len)

            read_id = example.header.read_id
            if len(read_id) > 36:
                raise ValueError(f'{read_id} is longer than 36 characters.')

            read_pos = example.header.pos

            examples[i] = (read_id, read_pos, fpos)

    df = pl.LazyFrame(examples)
    return df


def emit_examples(df: pl.LazyFrame, 
                  rf: Path, 
                  out_rf: Path, 
                  out_labels: Path, 
                  seq_len: int=31) -> int:
    with rf.open('rb') as src, out_rf.open('wb') as f_rf:
        labels = []

        df = df.select(['fpos', 'label']).collect()

        header = RFHeader.parse_header(src)
        header.n_examples = len(df)
        f_rf.write(header.to_bytes())

        for fpos, label in tqdm(df.iter_rows(), total=len(df)):
            src.seek(fpos)
            example = RFExample.from_file(src, seq_len)
            
            example = example.header.to_bytes() + example.data.to_bytes()
            f_rf.write(example)

            labels.append(label)

        labels = np.array(labels, dtype=np.half)
        with out_labels.open('wb') as out:
            out.write(labels.tobytes())

        return len(df)

def main(args: argparse.Namespace):
    pl.enable_string_cache()

    bedgraph = parse_bedgraph(args.bedgraph, args.min_cov, args.include_ctgs, args.exclude_ctgs)
    mappings = parse_mappings(args.mappings)

    tqdm.write('Extracting data information from rf file...')
    rf_data = rf_extract_info(args.rf, args.seq_len)

    tqdm.write('Mapping examples to labels...')
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
    
    tqdm.write('Writing data and labels...')
    total = emit_examples(df, args.rf, args.output_rf, args.output_labels, args.seq_len)

    tqdm.write(f'{total} examples written.')
    

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

    parser.add_argument('--seq_len', type=int, default=31)

    parser.add_argument('--output_rf', type=Path, required=True)
    parser.add_argument('--output_labels', type=Path, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)