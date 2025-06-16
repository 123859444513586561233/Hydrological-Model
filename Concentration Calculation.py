import datetime
import os.path
import traceback
from concurrent import futures
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
        # os.path.join('data', 'jz-r-20080625.xlsx'),
        dtype={'Year': str, 'Month': str, 'Day': str, 'Hour': str}
    )
    really = dict()
    for _, row in df.iterrows():
        really[''.join(map(lambda x: x.zfill(2), row.tolist()[:4]))] = row['Q']
    return really


class MyProblem(ea.Problem):  # 继承Problem父类
    REALLY = None

    def __init__(self, runoff_item, date_str, idx, folder_name):
        count = 7  # 决策变量维数
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 目标维数
        maxormins = [1]  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
        varTypes = [0] * count  # 决策变量的类型列表，0：实数；1：整数
        # cg, ci,cs, csat m, miu, tp
        lb = [0, 0, 0, 0, 0, 0, 0]  # 决策变量下界
        ub = [1, 1, 1, 1, 50, 50, 15]  # 决策变量上界
        self.runoff_item = runoff_item
        self.date_str = date_str
        if not self.REALLY:
            self.REALLY = get_real(folder_name)
        self.really = self.REALLY[date_str]
        self.hour_idx = idx
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, count, varTypes, lb, ub, lbin=[0] * count, ubin=[0] * count)

    def evalVars(self, v):  # 目标函数
        result = []
        for row in range(v.shape[0]):
            q_v = self.runoff_item.get_q(
                self.date_str, v[row][0], v[row][1], v[row][2], v[row][3],
                v[row][4], v[row][5], v[row][6], self.hour_idx
            )
            result.append([abs(self.really - sum(q_v)) / self.really])
        return np.array(result)


def miao(date: str, idx: int, result: List, folder: Path, output: str):
    logger.info(f'开始 | date={date}')
    runoff_day = RunoffDay(folder)
    problem = MyProblem(runoff_day, date, idx, folder.parts[-2])
    # 构建算法
    algorithm = ea.soea_DE_currentToBest_1_bin_templet(
        problem,
        ea.Population(
            Encoding='RI',
            NIND=5000,
            Field=ea.crtfld(
                'RI', problem.varTypes, problem.ranges,
                problem.borders, precisions=[30, 30, 30, 30, 4, 4, 4]
            )
        ),
        MAXGEN=30,  # 最大进化代数。
        logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        trappedValue=1e-10,  # 单目标优化陷入停滞的判断阈值。0
        maxTrappedCount=10  # 进化停滞计数器最大上限值。
    )
    # 求解
    try:
        res = ea.optimize(
            algorithm, verbose=True, drawing=0, outputMsg=True,
            drawLog=True, saveFlag=True, dirName=os.path.join(output, date)
        )
        print(res)
        if res['success']:
            var = list(res['Vars'][0])
            result.append(
                [folder.parts[-1], date, res['ObjV'][0][0]] + list(res['Vars'][0]) +
                runoff_day.get_q(date, var[0], var[1], var[2], var[3], var[4], var[5], var[6], idx, save=True) + [
                    res['executeTime']]
            )
        else:
            logger.warning(f'求解失败 date={date}')
            result.append([folder.parts[-1], date])
    except Exception as e:
        print(traceback.print_exc())


def solution(folder, now):
    result = list()  # 创建共享列表
    output = os.path.join('result-q', folder.parts[-1], now)
    for idx, folder in tqdm(list(enumerate(sorted(
            [i for i in folder.glob('*') if i.is_dir()],
            key=lambda x: x.parts[-1]
    ))), desc=folder.parts[-1]):
        miao(folder.parts[-1], idx, result, folder, output)
    pd.DataFrame(list(result)).to_excel(Path(output) / 'result.xlsx', index=False)


def main():
    now_data = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
    logger.add(Path('log') / f'{now_data}.log')
    tasks = []
    with futures.ProcessPoolExecutor(6) as t:
        for folder in Path('output1').glob('*'):
            if folder.is_file():
                continue
            tasks.append(t.submit(solution, folder, now_data))
        for task in futures.as_completed(tasks):
            try:
                task.result()
            except:
                print(traceback.print_exc())


if __name__ == '__main__':
    main()
