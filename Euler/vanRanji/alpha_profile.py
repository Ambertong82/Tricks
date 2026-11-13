"""固定站位 alpha 垂向剖面 (Van Ranji 冲刷算例, 2D)
====================================================
提取每个时间下 x/H = 4, 10, 20, 40 四个站位的 alpha.a 垂向剖面.

  - H = 0.25 m, 四个站位对应 x = 1.0 / 2.5 / 5.0 / 10.0 m
  - 站位取最近的网格列 (dx = 0.005 m), 实际用到的 x 打印在终端
  - 只输出 y > 0 的层 (床面以下不要)
  - 二维算例 z 只有一层, 不需要展向平均

输出: 一个时间一个 CSV (alpha_t40.csv, alpha_t45.csv, ...)
      列为 y, xH4, xH10, xH20, xH40
"""

import os

import fluidfoam as fl
import numpy as np

sol = "/media/amber/Elements/Scour/vanRanji/case0919_7coarse"
# sol = "/media/amber/Elements/Scour/vanTC/case1002_2"
# output_dir = "/home/amber/postpro/Scour/vanTC/alpha_profile_6"
output_dir = "/home/amber/postpro/Scour/vanRanji/alpha_profile_7"

times = np.arange(120, 130, 10)       # 算例里已有的时间目录
H = 0.25                             # 特征长度 (m)
x_over_H = [4, 10, 20, 40]           # 站位
y_min = 0.0                          # 只取 y > y_min 的层

# ---- 1. 提取网格坐标 ----
# X, Y, Z 的长度都是 580800
X, Y, Z = fl.readmesh(sol)

# ---- 2. 定义“精准抠取一列”的辅助函数 ----
def extract_column(X_field, Y_field, data_field, x_target, y_min_val, tol=1e-4):
    """
    通过距离掩码提取特定 x 站位的垂直剖面
    """
    # 找到网格中离目标 x 最近的真实 x 坐标
    idx_closest = np.argmin(np.abs(X_field - x_target))
    x_exact = X_field[idx_closest]
    
    # 获取这一整列的掩码 (tol=1e-4 完美包容了浮点数误差)
    col_mask = (np.abs(X_field - x_exact) < tol) & (Y_field > y_min_val)
    
    y_col = Y_field[col_mask]
    val_col = data_field[col_mask]
    
    # 按照 y 坐标从底部到顶部排序
    sort_idx = np.argsort(y_col)
    return y_col[sort_idx], val_col[sort_idx]

os.makedirs(output_dir, exist_ok=True)
header = "y," + ",".join(f"xH{xh:g}" for xh in x_over_H)

# ---- 3. 逐时间步提取数据并保存 ----
for t in times:
    # 读取 alpha.a 标量场
    alpha = fl.readscalar(sol, f"{float(t):g}", "alpha.a")
    
    columns_data = []
    max_alphas = []
    
    for i, xh in enumerate(x_over_H):
        x_target = xh * H
        
        # 提取该站位的 Y 坐标和 alpha 数据
        y_col, alpha_col = extract_column(X, Y, alpha, x_target, y_min)
        
        # 如果是第一个站位，把它的 y 坐标作为 CSV 的第一列基准
        if i == 0:
            columns_data.append(y_col)
            
        columns_data.append(alpha_col)
        max_alphas.append(f"x/H={xh:g}: {alpha_col.max():.5g}")
    
    # 将列表拼成矩阵并保存为 CSV
    try:
        table = np.column_stack(columns_data)
    except ValueError as e:
        # 如果报错，说明不同站位的 Y 网格层数不一致 (例如床面有起伏导致网格不规则)
        print(f"Error: 站位间的网格层数不一致！详情: {e}")
        continue

    out_csv = os.path.join(output_dir, f"alpha_t{float(t):g}.csv")
    np.savetxt(out_csv, table, delimiter=",", fmt="%.10g", header=header, comments="")
    
    print(f"t={float(t):g}: alpha max = " + ", ".join(max_alphas) + f"  -> {out_csv}")