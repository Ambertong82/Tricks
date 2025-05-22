import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# 1. 生成示例数据（替换为你的实际数据）
file1_path = "/home/amber/postpro/case090429_1_1e5.csv"
file2_path = "/home/amber/postpro/case230427_4_1e5.csv"
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

t1 = df1['Time'][0:]
t2 = df2['Time'][0:]
u1 = df1['U.b:0'][0:]
u2 = df2['U.a:0'][0:]

# 可视化
plt.scatter(t1, u1, label='$d=9\mu m$')
plt.scatter(t2, u2, label='$d=23\mu m$')
plt.xlabel('Time (t)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.savefig('/home/amber/postpro/velocity.png')  # 保存图像
plt.show()