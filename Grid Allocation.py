from concurrent import futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from loguru import logger
import rasterio
from rasterio.warp import calculate_default_transform

ROOT_PATH = Path(r'') 


def get_category_loc():
    with rasterio.open(Path(r'')) as src: 
             dataset_array = src.read(1)  
        label = sorted(set(dataset_array.flatten()))[:-1]

               lon_list = []
        lat_list = []
        row_list = []
        col_list = []
        for row in tqdm(range(src.height)):
            for col in range(src.width):
                if dataset_array[row, col] not in label:
                    continue
              
                x, y = src.xy(row, col)
                row_list.append(row)
                col_list.append(col)
               
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
    for f in ROOT_PATH.glob('*'): 
        logger.info(f'开始 file={f.parts[-1]}')
        main(f)
