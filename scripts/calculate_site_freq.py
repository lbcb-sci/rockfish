import polars as pl

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import argparse

from typing import *

N_NOMOD_COL = (pl.col('prob') <= 0.5).sum().alias('n_nomod')
N_MOD_COL = (pl.col('prob') > 0.5).sum().alias('n_mod')
FREQ_COL = (pl.col('n_mod') /
            (pl.col('n_nomod') + pl.col('n_mod'))).alias('freq')
END_COL = (pl.col('pos') + 1).cast(pl.UInt32).alias('end')


def parse_rockfish(path, get_predicted=False, type_suffix=''):
    rockfish = pl.scan_csv(path, has_header=False, sep='\t')

    if get_predicted:
        value_col = pl.when(pl.col('column_4') > 0.5).then(1).otherwise(
            0).alias(f'Rockfish{type_suffix}').cast(pl.UInt8)
    else:
        value_col = pl.col('column_4').alias(f'prob{type_suffix}').cast(
            pl.Float32)

    rockfish = rockfish.select([
        pl.col('column_1').alias('read_id'),
        pl.col('column_2').alias('ctg').cast(pl.Categorical),
        pl.col('column_3').alias('pos').cast(pl.UInt32), value_col
    ])

    return rockfish


def main(args):
    read_level = parse_rockfish(args.input)
    read_level = read_level.with_column((pl.col('prob') - 0.5).abs().alias('abs')) \
                           .sort('abs', reverse=True) \
                           .groupby(['ctg', 'pos']) \
                           .head(n=1000) \
                           .groupby(['ctg', 'pos']) \
                           .agg([N_MOD_COL, N_NOMOD_COL]) \
                           .sort(['ctg', 'pos']).collect()

    read_level = read_level.select([
        'ctg',
        pl.col('pos').alias('start'), END_COL, FREQ_COL, 'n_mod', 'n_nomod'
    ])

    read_level.write_csv(args.output, has_header=False, sep='\t')


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=Path)
    parser.add_argument('-o',
                        '--output',
                        type=Path,
                        default='site_freq.bedGraph')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    main(args)