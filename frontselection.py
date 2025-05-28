import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# 1. 生成示例数据（替换为你的实际数据）
# file1_path = "/home/amber/postpro/case090429_1_1e5.csv"
file_list = [
    '/home/amber/postpro/rawdata/case230427_4_{}.csv'.format(i) for i in range(23, 24)]
# df1 = pd.read_csv(file1_path)

all_results = pd.DataFrame()
for file in file_list:
    df = pd.read_csv(file)

    filtered_df = df[(df['alpha.a'] > 1e-5) & (df['Points:1'] > 0)]
    for A in [1 / 4, 1 / 3, 1 / 2]:
    # filtered_df=df[(df['alpha.a']>0.00001)]
        if len(filtered_df) > 1:

            front_pos = filtered_df['Points:0'].max()
            selected_point1 = front_pos - A * 0.3
    # print(front_pos)
            closest_point = (filtered_df['Points:0'] - selected_point1).abs().idxmin()
            x = filtered_df.loc[closest_point, 'Points:0']
            t = filtered_df.loc[closest_point, 'Time']
            result = filtered_df.loc[closest_point, [
            'Time', 'Points:0', 'Points:1', 'alpha.a', 'U.a:0', 'U.b:0']]
        # time=filtered_df.loc[max_point,'Time']
            print(f"x={A:.2f}front ", x, t)
