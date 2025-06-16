from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
import scipy.interpolate as interp


def cubic_spline_interpolation(arr, tasks):
    for task in tasks:
        number = len(task)
        arr[task[0]: task[-1] + 1] = interp.CubicSpline(np.array([0, 0 + number + 1]),
                                                        np.array([arr[task[0] - 1], arr[task[-1] + 1]]))(
            np.arange(0 + 1, number + 1))
    return arr


for folder in Path('output1').glob('*'): 
    logger.info(f'开始 | folder={folder}')
    date_list = list()
    df = pd.read_csv(folder / '1_.csv')
    mapping = pd.read_csv(folder / 'mapping.csv')
    df['year'] = df['name'].map(lambda x: int(x.replace('rainfall_idw_', '')[:4]))
    df['month'] = df['name'].map(lambda x: int(x.replace('rainfall_idw_', '')[4:6]))
    df['day'] = df['name'].map(lambda x: int(x.replace('rainfall_idw_', '')[6:8]))
    df['hour'] = df['name'].map(lambda x: int(x.replace('rainfall_idw_', '')[9:11]))
    df.sort_values(['year', 'month', 'day', 'hour'], inplace=True)
    df[['year', 'month', 'day', 'hour']].drop_duplicates().to_csv(folder / '2_.csv', index=False)
    df = pd.merge(df, mapping, on=['category_lon', 'category_lat'], how='left')
    # np.save(folder / 'category_row.npy', df['category_row'].values)
    # np.save(folder / 'category_col.npy', df['category_col'].values)
    np.save(
        folder / 'category_max_min_row.npy',
        np.array([df['category_row'].values.min(), df['category_row'].values.max()])
    )
    np.save(
        folder / 'category_max_min_col.npy',
        np.array([df['category_col'].values.min(), df['category_col'].values.max()])
    )
    rain_arr = list()
    for name, item in tqdm(df.groupby('name'), desc=f'解析降水量 {folder.parts[-1]}'):
        assert isinstance(name, str)
        arr = pd.pivot_table(item, columns='category_col', index='category_row', values='pixel_value').values
        arr = np.where(arr < 0, 0, arr)
        rain_arr.append(arr)
        date = name.replace('.tif', '').strip('rainfall_idw_').replace('_', '')
        (folder / date).mkdir(exist_ok=True, parents=True)
        date_list.append(date)
        np.save(folder / date / 'rain_arr.npy', arr)
    rain_arr = np.array(rain_arr)
  
  

    hour, height, width = rain_arr.shape
    bool_res = np.zeros(shape=rain_arr.shape)
    for row in tqdm(range(height), desc=f'是否强降水 {folder.parts[-1]}'):
        for col in range(width):
            point = rain_arr[:, row, col]
            if all(np.isnan(point)):
                bool_res[:, row, col] = np.array([False] * point.shape[0])
                continue
            
            bool_1 = np.array([False] * point.shape[0])  
            bool_2 = np.array([False] * point.shape[0])  
            bool_3 = (np.roll(point, shift=1) + point) == 0  
            bool_3[0] = False
            for idx, v in enumerate(point):
               
                if v > 20:
                    bool_1[idx] = True
                    continue
            
                if (idx >= 2) and (point[idx + 1 - 3: idx + 1].sum() > 50):
                    bool_2[idx] = True
                    continue
            
                if (idx >= 12) and (point[idx + 1 - 12: idx + 1].sum() > 90):
                    bool_2[idx] = True
                    continue
              
                if (idx >= 24) and (point[idx + 1 - 24: idx + 1].sum() > 150):
                    bool_2[idx] = True
                    continue
            bool_res[:, row, col] = bool_1 | (bool_2 & bool_3)
    for idx, date in enumerate(tqdm(date_list, desc=f'保存强降水 {folder.parts[-1]}')):
        np.save(folder / date / f'is_big_rain.npy', bool_res[idx, :, :])
