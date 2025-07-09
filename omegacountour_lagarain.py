import pandas as pd
import numpy as np

def find_and_separate_two_closest_points(file1_path, file2_path, output_path, target):
    # 读取第一个CSV文件（包含omega数据）
    try:
        df1 = pd.read_csv(file1_path, header=None)
    except Exception as e:
        print(f"Error reading first CSV file: {e}")
        return

    if len(df1) < 3:
        print("First CSV file needs at least 3 rows of data")
        return

    # 转换数据为数值类型
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    
    time_row = df1.iloc[0]  # 第一行是时间
    x_positions = df1.iloc[1]  # 第二行是x位置
    omega_data = df1.iloc[2:]  # 第三行开始是omega数据

    # 读取第二个CSV文件（包含y坐标数据）
    try:
        df2 = pd.read_csv(file2_path, header=None)
        df2 = df2.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"Error reading second CSV file: {e}")
        return

    results = []
    
    # 遍历每一列
    for col_idx in omega_data.columns:
        omega_series = omega_data[col_idx].dropna()
        if omega_series.empty:
            continue
            
        y_series = df2.iloc[2:, col_idx].dropna()
        
        # 确保两个系列有相同的索引
        common_index = omega_series.index.intersection(y_series.index)
        omega_series = omega_series[common_index]
        y_series = y_series[common_index]
        
        if omega_series.empty:
            continue
            
        # 计算与目标值的绝对距离
        distances = (omega_series - target).abs()
        
        # 获取距离最近的两个点
        closest_indices = distances.nsmallest(2).index
        
        # 获取这两个点的所有信息
        points = []
        for idx in closest_indices:
            points.append({
                'omega_value': omega_series[idx],
                'y_value': y_series[idx],
                'row_index': idx,
                'distance': distances[idx]
            })
        
        # 按照y值排序这两个点
        points_sorted = sorted(points, key=lambda x: x['y_value'])
        
        # 准备结果行
        result_row = {
            'column_index': col_idx,
            'time': time_row[col_idx] if pd.notna(time_row[col_idx]) else np.nan,
            'x_position': x_positions[col_idx] if pd.notna(x_positions[col_idx]) else np.nan,
        }
        
        # 添加第一个点的信息（y值较小的点）
        if len(points_sorted) > 0:
            result_row.update({
                'first_point_omega': points_sorted[0]['omega_value'],
                'first_point_y': points_sorted[0]['y_value'],
                'first_point_row': points_sorted[0]['row_index'],
                'first_point_distance': points_sorted[0]['distance'],
            })
        
        # 添加第二个点的信息（y值较大的点）
        if len(points_sorted) > 1:
            result_row.update({
                'second_point_omega': points_sorted[1]['omega_value'],
                'second_point_y': points_sorted[1]['y_value'],
                'second_point_row': points_sorted[1]['row_index'],
                'second_point_distance': points_sorted[1]['distance'],
            })
        if len(points_sorted) > 2:
            result_row.update({
                'three_point_omega': points_sorted[2]['omega_value'],
                'three_point_y': points_sorted[2]['y_value'],
                'three_point_row': points_sorted[2]['row_index'],
                'three_point_distance': points_sorted[2]['distance'],
            })
        
        results.append(result_row)

    if not results:
        print("No valid data points found")
        return

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 按照列索引排序
    results_df = results_df.sort_values('column_index')
    
    # 输出结果
    print(f"Three closest points to target value {target}, separated into columns:")
    print(results_df)
    
    # 保存结果到CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df

# 使用示例
if __name__ == "__main__":
    file1_path = "/home/amber/postpro/case230427_4_x14xomegaz.csv"  # 包含omega数据的文件
    file2_path = "/home/amber/postpro/case230427_4_x14xyy.csv"  # 包含y坐标数据的文件
    output_path = "/home/amber/postpro/mixinglayer/separated_closest_points.csv"
    target = 0.1  # 可以修改为你需要的目标值
    
    results = find_and_separate_two_closest_points(file1_path, file2_path, output_path, target)