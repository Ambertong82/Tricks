import numpy as np
import os
import pandas as pd
import fluidfoam

BASE_PATH = '/home/amber/postpro/selecting_variant/'
FILE_PREFIX = 'case230307_5'
sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/NEW/case230307_5"

# 时间步范围 - 两种表示方式
# 方式1: 直接使用时间步数字（转为字符串）
time_steps = [str(i) for i in range(4, 15, 1)]  # ['4', '5', '6', ..., '14']

# 或者方式2: 如果有特定的时间步命名
# time_steps = ["0.01", "0.02", ...]  # 根据实际的时间步目录名

A = 1/3
y_min = 1e-8
alpha_threshold = 1e-4

# 读取网格（只需要读一次）
print("Reading mesh...")
X, Y, Z = fluidfoam.readmesh(sol)

all_data = []  # 存储所有时间步的数据


def save_transposed_profiles(data_dict, base_path, file_prefix):
    output_files = {
        'yy': f"{file_prefix}_yy.csv",
        'ua': f"{file_prefix}_ua.csv",
    }

    for key, filename in output_files.items():
        data = []
        for item in data_dict:
            row = [item['time'], item['x']] + item[key]
            data.append(row)

        df = pd.DataFrame(data).transpose()
        output_file = f"{base_path}{filename}"
        df.to_csv(output_file, index=False, header=False)
        print(f"✓ Data saved to: {output_file} (shape={df.shape})")

for time_name in time_steps:
    print(f"Processing time {time_name}...")
    
    try:
        # 读取当前时间步的数据 - 注意: time_name 必须是字符串
        Ua = fluidfoam.readvector(sol, time_name, "U.a")
        Ub = fluidfoam.readvector(sol, time_name, "U.b")
        alpha_a = fluidfoam.readscalar(sol, time_name, "alpha.a")
        
    except Exception as e:
        print(f"  Error reading data at time {time_name}: {e}")
        # 检查该时间步目录是否存在
        time_dir = os.path.join(sol, time_name)
        if not os.path.exists(time_dir):
            print(f"  Directory {time_dir} does not exist. Skipping...")
        continue
    
    # 1. 找到 head_x（alpha > threshold 的最左侧x）
    # alpha_a[0] 是 x 方向的 alpha 值吗？还是整体 alpha？
    # 根据 fluidfoam 的输出结构，可能需要调整索引
    mask_head = (Y >= y_min) & (alpha_a > alpha_threshold)
    head_x = None
    for x in np.unique(X):
        mask = (X == x) & (Y >= y_min) & (alpha_a > alpha_threshold) & (Z==0.135)
        if np.any(mask):
            head_x = x
    if head_x is None:
        print(f"Warning: No head found at t={time_name}")
        continue
   

    
    # 2. 确定目标x位置
    target_x = head_x - A * 0.3
    unique_x = np.unique(X)
    closest_x = unique_x[np.argmin(np.abs(unique_x - target_x))]
    
    print(f"  head_x = {head_x:.4f}, target_x = {target_x:.4f}, using x = {closest_x:.4f}")
    
    # 3. 提取该x位置的整条y向剖面数据（与提供代码一致）
    mask = (X == closest_x) & (Y >= 0) & (Z == 0.135)

    if np.sum(mask) == 0:
        print(f"  Warning: No valid points found at x={closest_x:.4f} for t={time_name}")
        continue

    yvalue = Y[mask]
    xvalue = X[mask]
    uavalue = Ua[0][mask]

    if len(yvalue) == 0 or len(uavalue) == 0:
        print(f"  Warning: Empty extracted values at x={closest_x:.4f} for t={time_name}")
        continue

    # 按y排序，保证剖面顺序一致
    sort_idx = np.argsort(yvalue)
    yvalue = yvalue[sort_idx]
    xvalue = xvalue[sort_idx]
    uavalue = uavalue[sort_idx]

    # 4. 组织数据（与save_data转置格式一致）
    time_float = float(time_name)
    all_data.append({
        'time': time_float,
        'x': float(closest_x),
        'yy': yvalue.tolist(),
        'ua': uavalue.tolist(),
    })

# 5. 合并数据并保存
if all_data:
    save_transposed_profiles(all_data, BASE_PATH, FILE_PREFIX)

    total_points = sum(len(item['yy']) for item in all_data)
    print(f"\n✓ Total profile points: {total_points}")
    print(f"  Time steps processed: {len(time_steps)}")
    print(f"  Columns (time steps): {len(all_data)}")
    
    # 显示基本信息
    all_times = [item['time'] for item in all_data]
    all_y = np.concatenate([np.asarray(item['yy']) for item in all_data])
    all_x = [item['x'] for item in all_data]
    print("\nData summary:")
    print(f"  Time range: {min(all_times):.1f} to {max(all_times):.1f}")
    print(f"  y-range: [{all_y.min():.6f}, {all_y.max():.6f}]")
    print(f"  x positions used: {np.unique(np.asarray(all_x))}")
    
else:
    print("✗ No data was processed.")
    print("  Check that:")
    print("  1. The case directory exists")
    print("  2. Time step directories exist (e.g., '4', '5', etc.)")
    print("  3. The field files exist in time directories")
