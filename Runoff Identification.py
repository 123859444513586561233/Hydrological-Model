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


for folder in Path('output1').glob('*'):  # todo 单场洪水编号
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
    #
    # # 插值
    # min_arr = rain_arr[0].copy()
    # height, width = rain_arr[0].shape
    # for i in tqdm(range(height)):
    #     for j in range(width):
    #         if np.isnan(min_arr[i, j]):
    #             continue
    #         # 提取当前位置的所有值
    #         values = rain_arr[:, i, j]
    #         non_negative_values = values[values > 0]
    #         # 如果存在非负值，取最小值；否则保留 NaN
    #         if non_negative_values.size > 0:
    #             min_arr[i, j] = np.min(non_negative_values)
    #         else:
    #             min_arr[i, j] = 0
    # if np.nansum(rain_arr[0, :, :]) == 0:
    #     rain_arr[0, :, :] = min_arr
    # if np.nansum(rain_arr[-1, :, :]) == 0:
    #     rain_arr[-1, :, :] = min_arr
    #
    # seq = None
    # tasks = list()
    # for idx in range(rain_arr.shape[0]):
    #     v = np.nansum(rain_arr[idx, :, :])
    #     if v == 0:
    #         # 第一次
    #         if seq is None:
    #             seq = [idx]
    #         else:
    #             seq.append(idx)
    #     else:
    #         if isinstance(seq, list):
    #             tasks.append(seq)
    #             seq = None
    # if isinstance(seq, list):
    #     tasks.append(seq)
    #
    # cubic_rain_arr = np.zeros(shape=rain_arr.shape)
    # hour, height, width = rain_arr.shape
    # for row in tqdm(range(height), desc=f'插值'):
    #     for col in range(width):
    #         v = rain_arr[:, row, col]
    #         if np.isnan(v).all():
    #             cubic_rain_arr[:, row, col] = np.nan
    #         else:
    #             cubic_rain_arr[:, row, col] = cubic_spline_interpolation(v, tasks)
    #
    # for idx in range(cubic_rain_arr.shape[0]):
    #     (folder / date_list[idx]).mkdir(exist_ok=True, parents=True)
    #     np.save(folder / date_list[idx] / 'rain_arr.npy', cubic_rain_arr[idx])

    hour, height, width = rain_arr.shape
    bool_res = np.zeros(shape=rain_arr.shape)
    for row in tqdm(range(height), desc=f'是否强降水 {folder.parts[-1]}'):
        for col in range(width):
            point = rain_arr[:, row, col]
            if all(np.isnan(point)):
                bool_res[:, row, col] = np.array([False] * point.shape[0])
                continue
            # 参考相关文献中不同历时强降雨的定义方法，将降雨事件中1h内的降雨量达到20mm，持续3h内的累积降雨量大于50mm，
            # 持续12h内的降雨量大于90mm，以及持续24h内的累积降雨量大于150mm的情况定义为强降雨，
            # 这里需要特别说明的是如果持续的时间里有中间大于等于2h的降雨均为0的情况不算，这种情况下可以看作是新的降雨事件。
            # 这些时刻对应的主导产流机制类型为图1a，否者对应的主导产流机制类型为图1b。不同降雨条件下的主导产流机制辨析见图1
            bool_1 = np.array([False] * point.shape[0])  # 将降雨事件中1h内的降雨量达到20mm，
            bool_2 = np.array([False] * point.shape[0])  # 持续x h内的累积降雨量大于x  mm，
            bool_3 = (np.roll(point, shift=1) + point) == 0  # 这里需要特别说明的是如果持续的时间里有中间大于等于2h的降雨均为0的情况不算
            bool_3[0] = False
            for idx, v in enumerate(point):
                # 将降雨事件中1h内的降雨量达到20mm，
                if v > 20:
                    bool_1[idx] = True
                    continue
                # 持续3h内的累积降雨量大于50mm，
                if (idx >= 2) and (point[idx + 1 - 3: idx + 1].sum() > 50):
                    bool_2[idx] = True
                    continue
                # 持续12h内的降雨量大于90mm
                if (idx >= 12) and (point[idx + 1 - 12: idx + 1].sum() > 90):
                    bool_2[idx] = True
                    continue
                # 以及持续24h内的累积降雨量大于150mm的情况定义为强降雨，
                if (idx >= 24) and (point[idx + 1 - 24: idx + 1].sum() > 150):
                    bool_2[idx] = True
                    continue
            bool_res[:, row, col] = bool_1 | (bool_2 & bool_3)
    for idx, date in enumerate(tqdm(date_list, desc=f'保存强降水 {folder.parts[-1]}')):
        np.save(folder / date / f'is_big_rain.npy', bool_res[idx, :, :])
