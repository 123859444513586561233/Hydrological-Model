import traceback
from concurrent import futures
from pathlib import Path
from copy import deepcopy
from itertools import product

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import rasterio

from runoff_day import RunoffDay

CATEGORY_ORDER = ['qg', 'qi', 'qs', 'qsat']
PROCESS_AMOUNT = 30


def change_tif(tif_data, path, base_tif_df):
    tif_df = pd.DataFrame(tif_data, columns=['cate', 'row', 'col', 'v'])
    with rasterio.open('data/jz-raster-float64.tif', 'r') as src:
               profile = src.profile
    for idx, v in enumerate(CATEGORY_ORDER):
        with rasterio.open(path / f'{v}_mean.tif', 'w', **deepcopy(profile)) as dst:
            dst.write(pd.pivot(pd.merge(
                base_tif_df, tif_df[tif_df['cate'] == idx],
                on=['col', 'row'], how='left'
            )[['row', 'col', 'v']], columns='col', index='row', values='v').values, 1)

    with rasterio.open(path / f'all_q_mean.tif', 'w', **deepcopy(profile)) as dst:
        dst.write(pd.pivot(pd.merge(
            base_tif_df, tif_df,
            on=['col', 'row'], how='left'
        )[['row', 'col', 'v']], columns='col', index='row', values='v').values, 1)


def fun(folder, base_tif_df):
    runoff_day = RunoffDay(folder, is_log=False)
    category = runoff_day.get_category_idx()
    melt_list = list()
    tif_data = list()
    r_item = dict()
    for category_key, category_value in category.items():
        category_key = dict(Rs='qs', Rint='qi', Rsat='qsat', Rg='qg')[category_key]
        arr_path = folder / 'best_q' / f'{folder.parts[-1]}_{category_key}.npy'
        if not arr_path.exists():
            print(arr_path)
            continue
        arr = np.load(arr_path)
        r_item[category_key] = arr.mean()
        for (row, col), v in zip(category_value, arr):
            melt_list.append([row, col, v])
            tif_data.append([
                CATEGORY_ORDER.index(category_key),
                row + runoff_day.use_row[0],
                col + runoff_day.use_col[0], v
            ])

    change_tif(tif_data, folder, base_tif_df)


def miao(path):
    height, width = RunoffDay.DATA_ARR.shape
    base_tif_df = pd.DataFrame(product(range(height), range(width)), columns=['row', 'col'])
    folders = [folder for folder in path.glob('*') if folder.is_dir()][:]

    tasks = []
    with futures.ProcessPoolExecutor(PROCESS_AMOUNT) as t:
        for folder in tqdm(folders, desc=path.parts[-1]):
            tasks.append(t.submit(fun, folder, base_tif_df))
        for task in tqdm(futures.as_completed(tasks)):
            try:
                task.result()
            except:
                print(traceback.print_exc())


def run():
    for folder in list(Path('output1').glob('*')):
        if folder.is_file():
            continue
        logger.info(f'开始一个 | folder={folder}')
        miao(folder)


if __name__ == '__main__':
    run()
