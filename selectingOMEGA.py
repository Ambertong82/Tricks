import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
file_list = [
    f'/home/amber/postpro/depth_average/case230427_4_{i}.csv' for i in range(0, 5)]
# file_list=['alpha_1.csv','alpha_2.csv']
all_results = pd.DataFrame()
data_dict = []
data_dict2 = []
data_dict3 = []

A = 1/4

for file in file_list:
    df = pd.read_csv(file)
    # filtered_df=df[(df['alpha.a']>0.0005)&(df['alpha.a']<0.001)]
    front_pos = df['xx'].max()
    selected_point1 = front_pos - A * 0.3
    #print(front_pos)
    closest_point = (df['xx'] -selected_point1).abs().idxmin()
    x = df.loc[closest_point, 'xx']
    #print(x)
    h_depth = df.loc[(df['xx'] == x), 'h_depth']
    H_depth = df.loc[(df['xx'] == x),'H_depth']
    time  = df.loc[(df['xx'] == x), 'time']
    data_dict.append(time.tolist() + [x] + h_depth.tolist() + H_depth.tolist())
    #rint(data_dict)

data_df = pd.DataFrame(data_dict).transpose()
data_df.to_csv(
    '/home/amber/postpro/case230427_4_14.csv',
    index=False,
    header=False)









