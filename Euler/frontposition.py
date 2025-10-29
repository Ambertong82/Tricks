# import pandas as pd
# file_list = [
#     '/home/amber/postpro/rawdata/case0801_1_{}.csv'.format(i) for i in range(0, 79, 1)]
# # file_list=['alpha_1.csv','alpha_2.csv']
# all_results = pd.DataFrame()
# data_dict = []
# data_dict2 = []
# for file in file_list:
#     df = pd.read_csv(file)
#     #filtered_df=df[(df['alpha.a']>1e-4)&(df['alpha.a']<0.001)]
#     filtered_df = df[(df['alpha.saline'] > 1e-5) & (df['Points:1'] > 0)]
#     #filtered_df=df[(df['alpha.a']>5e-5) & (df['Points:1'] > 0)]
#     if len(filtered_df) > 1:
#         max_point = filtered_df['Points:0'].idxmax()
#         result = filtered_df.loc[max_point, [
#             'Time', 'Points:0', 'Points:1', 'alpha.saline', 'U:0']]
#         #result = filtered_df.loc[max_point, [
#             #'Time', 'Points:0', 'Points:1', 'alpha.a', 'U.a:0']]
#         # time=filtered_df.loc[max_point,'Time']
#         # print(time)
#         all_results = pd.concat([all_results, result.to_frame().T])
# all_results.to_csv('/home/amber/postpro/run7case0801_1.csv', index=False)


import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
#from fluidfoam import getTimeNames

#sol = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_6"
#sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
#sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090912_1"
# sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4fine"
sol = '/media/amber/53EA-E81F/PhD/case231020_5'
#sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
output_dir = "/home/amber/postpro/frontposition_turbidity"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'front_position_2310205mid.csv')

# 参数设置
alpha_threshold = 1e-5   # alpha.a 的头部阈值
y_min = 0                # 垂向积分下限（避免壁面影响）
times = np.arange(1, 40, 1)


X, Y, Z = fluidfoam.readmesh(sol)

# 创建CSV文件并写入表头
with open(output_file, 'w') as f:
    f.write('Time,Head_X,Ux,Uy,Uz\n')

for time_v in times:
    # 读取场数据
    Ua_A = fluidfoam.readvector(sol, str(time_v), "U.a")
    alpha_A = fluidfoam.readscalar(sol, str(time_v), "alpha.a")
    

    # --- 定位头部位置（alpha.a > alpha_threshold 的最大 x 坐标）---
    head_x = None

    # 找到所有满足alpha阈值的点
    valid_mask = (alpha_A > alpha_threshold) & (Y >= y_min) & (Z ==0.125)
    if not np.any(valid_mask):
        print(f"Warning: No head found at t={time_v}")
        continue

    # 从满足条件的点中找出x坐标最大的值
    head_x = np.max(X[valid_mask])

    # 找到x坐标等于head_x的所有点（可能有多个y坐标）
    head_indices = np.where((X == head_x) & valid_mask)[0]

    if len(head_indices) == 0:
        print(f"Warning: No head found at t={time_v}")
        continue

    # 取第一个满足条件的点（或根据需求调整，如取平均）
    head_idx = head_indices[0]

    

    #print(head_idx)
    head_x = head_x
    #print(head_x)
    U_front = Ua_A[:,head_idx]  # 直接索引速度分量
    

    # 写入CSV
    with open(output_file, 'a') as f:
        f.write(f"{time_v},{head_x},{U_front[0]},{U_front[1]},{U_front[2]}\n")
    
    print(f"At time {time_v}, head position: {head_x:.3f} m, Ux: {U_front[0]:.3f} m/s")

print(f"Front positions saved to {output_file}")