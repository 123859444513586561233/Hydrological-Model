from concurrent import futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from loguru import logger
import rasterio
from rasterio.warp import calculate_default_transform

ROOT_PATH = Path(r'F:\XYYpaper\paper3-DJ\GIS\yuecheng\DaiMa\RainIDW')  # todo 所有洪水的路径


def get_category_loc():
    with rasterio.open(Path(r'data\yc-raster.tif')) as src:  # todo 网格图层
        # 创建一个空数组用于存储转换后的数据
        dataset_array = src.read(1)  # 读取栅格数据
        label = sorted(set(dataset_array.flatten()))[:-1]

        # 遍历每个像素，转换为经纬度
        lon_list = []
        lat_list = []
        row_list = []
        col_list = []
        for row in tqdm(range(src.height)):
            for col in range(src.width):
                if dataset_array[row, col] not in label:
                    continue
                # 获取投影坐标
                x, y = src.xy(row, col)
                row_list.append(row)
                col_list.append(col)
                # 转换为经纬度
                lon, lat = rasterio.warp.transform(src.crs, 'EPSG:4326', [x], [y])
                lon_list.append(lon[0])
                lat_list.append(lat[0])
    return lon_list, lat_list, row_list, col_list


CATEGORY_LON_LIST, CATEGORY_LAT_LIST, CATEGORY_ROW_LIST, CATEGORY_COL_LIST = get_category_loc()


def run(file, all_data):
    src = rasterio.open(file)
    row, col = src.index(CATEGORY_LON_LIST, CATEGORY_LAT_LIST)
    pixel_value = src.read(1)[row, col]
    all_data.extend(list(zip([file.name] * len(row), CATEGORY_LON_LIST, CATEGORY_LAT_LIST, row, col, pixel_value)))


def main(folder):
    # 定义地理坐标系（WGS84）和目标投影坐标系（Albers）
    # wgs84 = pyproj.CRS("EPSG:4326")  # WGS84地理坐标系
    # src = rasterio.open(next(Path(folder).glob('rainfall_idw_*_*.tif')))
    # albers = pyproj.CRS.from_wkt(src.crs.to_string())

    # 创建投影转换器
    # transformer = pyproj.Transformer.from_crs(wgs84, albers, always_xy=True)
    # transform_lon, transform_lat = transformer.transform(CATEGORY_LON_LIST, CATEGORY_LAT_LIST)
    output = Path('output1') / folder.parts[-1]
    if not output.exists():
        output.mkdir(exist_ok=True, parents=True)
    if (output / 'mapping.csv').exists() and (output / '1_.csv').exists():
        return
    if not (output / 'mapping.csv').exists():
        pd.DataFrame(
            zip(CATEGORY_LON_LIST, CATEGORY_LAT_LIST, CATEGORY_ROW_LIST, CATEGORY_COL_LIST),
            columns=['category_lon', 'category_lat', 'category_row', 'category_col'],
        ).to_csv(output / 'mapping.csv', index=False)

    all_data = list()
    tasks = []
    with futures.ThreadPoolExecutor(16) as t:
        for file in tqdm(list(Path(folder).glob('rainfall_idw_*_*.tif'))):
            tasks.append(t.submit(run, file, all_data))
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            task.result()

    pd.DataFrame(
        all_data,
        columns=['name', 'category_lon', 'category_lat', 'row', 'col', 'pixel_value']
    ).to_csv(output / '1_.csv', index=False)


if __name__ == '__main__':
    for f in ROOT_PATH.glob('*'):  # todo: 单场
        logger.info(f'开始 file={f.parts[-1]}')
        main(f)
