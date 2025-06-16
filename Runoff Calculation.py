import datetime
import os.path
import traceback
from concurrent import futures
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List

import geatpy as ea
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

from runoff_day import RunoffDay


def get_real(date_str) -> Dict[str, float]:
    df = pd.read_excel(
        os.path.join('data', f'jz-r-{date_str}.xlsx'),
        # os.path.join('data', 'yc-r-20080625.xlsx'),
        dtype={'Year': str, 'Month': str, 'Day': str, 'Hour': str}
    )
    really = dict()
    for _, row in df.iterrows():
        really[''.join(map(lambda x: x.zfill(2), row.tolist()[:4]))] = row['r']
    return really


class MyProblem(ea.Problem): 
    REALLY = None

    def __init__(self, runoff_item, date_str, folder_name):
        count = 4
        name = 'MyProblem' 
        M = 1
        maxormins = [1]  
        varTypes = [0] * count  
        # 'fc', 'fp', 'fca', 'fcb'
        lb = [0] * count 
        # ub = [1.2, 7.8, 4.7, 3.6]  
        ub = [7.6, 41.4, 30.6, 28.7] 
        # ub = [1000]  * count 
        self.runoff_item = runoff_item
        self.date_str = date_str
        if not self.REALLY:
            self.REALLY = get_real(folder_name)
        self.really = self.REALLY[date_str]
     
        ea.Problem.__init__(self, name, M, maxormins, count, varTypes, lb, ub, lbin=[0] * count, ubin=[1] * count)

    def evalVars(self, v): 
        result = []
        for row in range(v.shape[0]):
            r_v = self.runoff_item.get_r(self.date_str, v[row][0], v[row][1], v[row][2], v[row][3])
            result.append([abs(self.really - sum(r_v)) / self.really])
            # 'fc', 'fp', 'fca', 'fcb'
            # fp >= fca >= fcb >= fc
            # fp >= fca
        return np.array(result), (
                np.where((v[:, 2] - v[:, 1]) < 0, 0, v[:, 2] - v[:, 1]) +
                np.where((v[:, 3] - v[:, 2]) < 0, 0, v[:, 3] - v[:, 2]) +
                np.where((v[:, 0] - v[:, 3]) < 0, 0, v[:, 0] - v[:, 3])
        ).reshape(v.shape[0], 1)


def miao(date: str, result: List, folder: Path, output: str):
    logger.info(f'开始 | date={date}')
    runoff_day = RunoffDay(folder)
    problem = MyProblem(runoff_day, date, folder.parts[-2])
 
    algorithm = ea.soea_DE_currentToBest_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=5000),
        MAXGEN=50, 
        logTras=0, 
        trappedValue=1e-6, 
        maxTrappedCount=6  
    )

    try:
        res = ea.optimize(
            algorithm, verbose=True, drawing=1, outputMsg=True,
            drawLog=False, saveFlag=True, dirName=os.path.join(output, date)
        )
        if res['success']:
            var = list(res['Vars'][0])
            result.append(
                [folder.parts[-1], date, res['ObjV'][0][0]] + list(res['Vars'][0]) +
                runoff_day.get_r(date, var[0], var[1], var[2], var[3], save=True) + [res['executeTime']]
            )
        else:
            logger.warning(f'求解失败 date={date}')
            result.append([folder.parts[-1], date])
    except Exception as e:
        print(traceback.print_exc())
    logger.info(f'完成 | date={date}')


def solution(folder, now):
    output = os.path.join('result', folder.parts[-1], now)
    manager = Manager()
    result = manager.list()  
    tasks = []
    with futures.ProcessPoolExecutor(30) as t:
        for folder in tqdm(list(folder.glob('*'))):
            if folder.is_file():
                continue
            tasks.append(t.submit(miao, folder.parts[-1], result, folder, output))
        for task in tqdm(futures.as_completed(tasks), total=len(tasks)):
            try:
                task.result()
            except:
                print(traceback.print_exc())

    # result = list()
    # for folder in tqdm(list(folder.glob('*'))):
    #     if folder.is_file():
    #         continue
    #     miao(folder.parts[-1], result, folder, output)
    pd.DataFrame(list(result)).to_excel(Path(output) / 'result.xlsx', index=False)


def main():
    for f in list(Path('output1').glob('*')):
        if f.is_file():
            continue
        logger.info(f'开始 {f}')
        now_data = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
        logger.add(Path('log') / f'{now_data}.log')
    
        try:
            solution(f, now_data)
        except:
            print(f'报错了 | f={f}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
    # miao('', [], Path(''), 'test')
