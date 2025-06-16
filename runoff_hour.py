import inspect
import traceback
from collections import defaultdict
import datetime
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from time import perf_counter


class MyList(list):
    def __setitem__(self, index, value):
        print(f"设置索引 {index} 的值为 {value}")
        log_file = Path('log') / f"{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}trace_log.txt"
        traceback.print_stack()
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"设置索引 {index} 的值为 {value}\n")
        
            for frame_info in inspect.stack():
                frame = frame_info.frame
                f.write(f"函数: {frame_info.function}  文件: {frame_info.filename}  行号: {frame_info.lineno}\n")
           
                for var, val in frame.f_locals.items():
                    f.write(f"    {var} = {val!r}\n")
                f.write("-" * 40 + "\n")
   
        super().__setitem__(index, value)


class RunoffDay(object):
    DATA_ARR = rasterio.open(Path('data/jz-raster.tif')).read(1)  #
    LOC = pd.read_excel('data/loc.xlsx') 

    def __init__(self, folder: Path, is_load_category_idx=True, is_log=False):
        if is_log:
            logger.info(f'开始初始化 RunoffDay | folder={folder.parts[-1]}')
        self.folder = folder
        self.best_r = self.folder / 'best_r'
        self.best_r_cubic = self.folder / 'best_r_allocate'
        self.best_q = self.folder / 'best_q'
        for f in [self.best_r, self.best_r_cubic, self.best_q]:
            f.mkdir(exist_ok=True)
        self.rain_arr = np.load(folder / 'rain_arr.npy')

        self.date = [''.join([str(i).zfill(2) for i in j]) for j in
                     pd.read_csv(folder.parent / '2_.csv').values.tolist()]
        self.use_col = np.load(folder.parent / 'category_max_min_col.npy')
        self.use_row = np.load(folder.parent / 'category_max_min_row.npy')
        if is_load_category_idx:
            self.is_big_rain = np.load(folder / 'is_big_rain.npy')
            self.category_idx = self.get_category_idx()
        self.area = ** 
        if is_log:
            logger.info(f'完成初始化 RunoffDay | folder={folder.parts[-1]}')
        self.r_cache = dict()
        self.q_cache = dict()
        self.file_not_found_log_msg = list()

    def get_raster_clean_arr(self):
        return self.DATA_ARR[
               self.use_row[0]:self.use_row[1] + 1,
               self.use_col[0]:self.use_col[1] + 1
               ]

    def get_category_idx(self):
        loc_mapping = {k: [v1, v2] for k, v1, v2 in zip(self.LOC['label_value'], self.LOC['Rmin'], self.LOC['Rmax'])}
        category_loc = defaultdict(list)
        height, width = self.is_big_rain.shape
        for row in range(height):
            for col in range(width):
                v = self.is_big_rain[row, col]
                if np.isnan(v):
                    continue
                category_value = self.get_raster_clean_arr()[row, col]
                if not category_value in loc_mapping:
                    continue
                category_loc[loc_mapping[category_value][1 if v else 0]].append(MyList([row, col]))
        return category_loc

    def get_r(
            self, date_str: str, fc: Union[int, float], fp: Union[int, float],
            fca: Union[int, float], fcb: Union[int, float], save=False
    ) -> List[Union[int, float]]:
        """
       
        """
        rg = 0
        rs = 0
        r_sat = 0
        r_int = 0
      
        if 'Rg' in self.category_idx and self.category_idx['Rg']:
            # arr = rain[[i[0] for i in loc['Rg']], [i[1] for i in loc['Rg']]]
            if 'Rg' in self.r_cache:
                arr = self.r_cache['Rg']
            else:
                arr = self.rain_arr[
                    [k[0] for k in [[i, j] for i, j in self.category_idx['Rg'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]],
                    [k[1] for k in [[i, j] for i, j in self.category_idx['Rg'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]]
                ]
            arr = np.where(arr >= fc, fc, arr)
            self.r_cache['Rg'] = arr
            if save:
                np.save(self.best_r / f'{date_str}_rg.npy', arr)
            rg = arr.mean()

       
        if 'Rs' in self.category_idx and self.category_idx['Rs']:
            if 'Rs' in self.r_cache:
                arr = self.r_cache['Rs']
            else:
                arr = self.rain_arr[
                    [k[0] for k in [[i, j] for i, j in self.category_idx['Rs'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]],
                    [k[1] for k in [[i, j] for i, j in self.category_idx['Rs'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]]
                ]
                self.r_cache['Rs'] = arr
            arr = np.where(arr > fp, arr - fp, 0)
            if save:
                np.save(self.best_r / f'{date_str}_rs.npy', arr)
            rs = arr.mean()

    
        if 'Rsat' in self.category_idx and self.category_idx['Rsat']:
            if 'Rsat' in self.r_cache:
                arr = self.r_cache['Rsat']
            else:
                arr = self.rain_arr[
                    [k[0] for k in [[i, j] for i, j in self.category_idx['Rsat'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]],
                    [k[1] for k in [[i, j] for i, j in self.category_idx['Rsat'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]]
                ]
                self.r_cache['Rsat'] = arr
            r_int = np.where((fcb < arr) & (arr <= fca), arr - fcb, np.where(arr > fca, fca - fcb, 0))
            arr = np.where(arr > (r_int + fcb), arr - (r_int + fcb), 0)
            if save:
                np.save(self.best_r / f'{date_str}_r_sat.npy', arr)
            r_sat = arr.mean()

    
        if 'Rint' in self.category_idx and self.category_idx['Rint']:
            if 'Rint' in self.r_cache:
                arr = self.r_cache['Rint']
            else:
                arr = self.rain_arr[
                    [k[0] for k in [[i, j] for i, j in self.category_idx['Rint'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]],
                    [k[1] for k in [[i, j] for i, j in self.category_idx['Rint'] if
                                    i < self.rain_arr.shape[0] and j < self.rain_arr.shape[1]]]
                ]
                self.r_cache['Rint'] = arr
            arr = np.where((fcb < arr) & (arr <= fca), arr - fcb, np.where(arr > fca, fca - fcb, 0))
            if save:
                np.save(self.best_r / f'{date_str}_r_int.npy', arr)
            r_int = arr.mean()

        return [rg, rs, r_sat, r_int]

    def load_q_arr(self, path):
        path_str = str(path)
        if path_str in self.q_cache:
            return self.q_cache[path_str]
        arr = np.load(path)
        self.q_cache[path_str] = arr
        return arr

    def my_file_not_fond_log(self, msg):
        if msg not in self.file_not_found_log_msg:
            self.file_not_found_log_msg.append(msg)
            logger.warning(msg)

    def get_q(
            self, date_str: str, cg: float, ci: float, cs: float, csat: float,
            m: float, miu: float, tp: float, idx: int, save=False
    ):
        qg_t_1 = 0
        qi_t_1 = 0
        qs_t_1 = 0
        qsat_t_1 = 0
        if idx > 0:
            last_data_str = self.date[idx - 1]
            parts = list(self.best_q.parts)
            parts[-2] = last_data_str
            base_path = Path(*parts)
            if (base_path / f'{last_data_str}_qg_sum.npy').exists():
                qg_t_1 = self.load_q_arr(base_path / f'{last_data_str}_qg_sum.npy')
            else:
                self.my_file_not_fond_log(f'文件不存在 | file={base_path}/{last_data_str}_qg_sum.npy')
            if (base_path / f'{last_data_str}_qi_sum.npy').exists():
                qi_t_1 = self.load_q_arr(base_path / f'{last_data_str}_qi_sum.npy')
            else:
                self.my_file_not_fond_log(f'文件不存在 | file={last_data_str}_qi_sum.npy')
            if (base_path / f'{last_data_str}_qs_sum.npy').exists():
                qs_t_1 = self.load_q_arr(base_path / f'{last_data_str}_qs_sum.npy')
            else:
                self.my_file_not_fond_log(f'文件不存在 | file={last_data_str}_qs_sum.npy')
            if (base_path / f'{last_data_str}_qsat_sum.npy').exists():
                qsat_t_1 = self.load_q_arr(base_path / f'{last_data_str}_qsat_sum.npy')
            else:
                self.my_file_not_fond_log(f'文件不存在 | file={last_data_str}_qsat_sum.npy')

        qg = 0
        if (self.best_r_cubic / f'{date_str}_rg.npy').exists():
            rg_arr = self.load_q_arr(self.best_r_cubic / f'{date_str}_rg.npy')
            arr = cg * qg_t_1 + (1 - cg) * rg_arr * (self.area / 3.6)
            if save:
                print(self.best_q / f'{date_str}_qg.npy')
                np.save(self.best_q / f'{date_str}_qg.npy', arr)

                np.save(self.best_q / f'{date_str}_qg_sum.npy', arr.sum())
            qg = arr.sum()
        else:
            self.my_file_not_fond_log(f'文件不存在 | file={date_str}_rg.npy')

        qi = 0
        if (self.best_r_cubic / f'{date_str}_r_int.npy').exists():
            r_int_arr = self.load_q_arr(self.best_r_cubic / f'{date_str}_r_int.npy')
            arr = ci * qi_t_1 + (1 - ci) * r_int_arr * (self.area / 3.6)
            if save:
                np.save(self.best_q / f'{date_str}_qi.npy', arr)
                np.save(self.best_q / f'{date_str}_qi_sum.npy', arr.sum())
            qi = arr.sum()
        else:
            self.my_file_not_fond_log(f'文件不存在 | file={date_str}_r_int.npy')

        qs = 0
        if (self.best_r_cubic / f'{date_str}_rs.npy').exists():
            rs_arr = self.load_q_arr(self.best_r_cubic / f'{date_str}_rs.npy')
            tmp1 = (1 + m * np.e ** (miu * (-1 - tp))) ** (-1 / m)
            tmp2 = (1 + m * np.e ** (miu * (-tp))) ** (-1 / m)
            arr = (1 - cs) * self.area * rs_arr * (tmp1 - tmp2) + cs * qs_t_1
            if save:
                np.save(self.best_q / f'{date_str}_qs.npy', arr)
                np.save(self.best_q / f'{date_str}_qs_sum.npy', arr.sum())
            qs = arr.sum()
        else:
            self.my_file_not_fond_log(f'文件不存在 | file={date_str}_rs.npy')

        qsat = 0
        if (self.best_r_cubic / f'{date_str}_r_sat.npy').exists():
            r_sat_arr = self.load_q_arr(self.best_r_cubic / f'{date_str}_r_sat.npy')
            tmp1 = (1 + m * np.e ** (miu * (- 1 - tp))) ** (-1 / m)
            tmp2 = (1 + m * np.e ** (miu * (- tp))) ** (-1 / m)
            arr = (1 - csat) * self.area * r_sat_arr * (tmp1 - tmp2) + qsat_t_1 * csat
            if save:
                np.save(self.best_q / f'{date_str}_qsat.npy', arr)
                np.save(self.best_q / f'{date_str}_qsat_sum.npy', arr.sum())
            qsat = arr.sum()
        else:
            self.my_file_not_fond_log(f'文件不存在 | file={date_str}_r_sat.npy')

        return [qg, qi, qs, qsat]


if __name__ == '__main__':
    runoff_item = RunoffDay(Path(''), True)
    for i in range(10):
        a = perf_counter()
        print(runoff_item.get_q(
            '2007060711', 9.31323E-10, 9.31323E-10,
            0.151428223, 0.999999999, 28.98710563, 38.9251152, 13.59286767, 30, save=False))
        print(perf_counter() - a)
