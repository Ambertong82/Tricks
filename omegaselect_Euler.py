import pandas as pd
import numpy as np
from pathlib import Path
from fractions import Fraction

# 常量定义

BASE_PATH = Path('/home/amber/postpro/')
RAW_DATA_DIR = BASE_PATH / 'rawdata'
OUTPUT_DIR = BASE_PATH / 'mixinglayerEuler'
FILE_PREFIX = 'case230427_4'

def get_output_filename(input_file):
    """生成对应每个输入文件的输出文件名"""
    # 从输入文件名提取编号（如 case230427_4_5.csv → 5）
    file_index = Path(input_file).stem.split('_')[-1]
    #frac = Fraction(a_value).limit_denominator()
    #a_id = f"{frac.numerator}{frac.denominator}"
    return f"{FILE_PREFIX}_output_{file_index}.csv"

def get_front_position(df):
    """从文件中获取前沿位置"""
    # 读取数据

    filtered_df = df[(df['alpha.a'] > 1e-5) & (df['Points:1'] > 0)]
    
    if len(filtered_df) > 1:

        front_pos = filtered_df['Points:0'].max()
        return front_pos
    return np.nan

def calculate_omegaz(df):
    """向量化计算涡量"""
    return df['grad(U.b):1'] - df['grad(U.b):3']

# def detect_sign_changes(omega_z, y_values, x_values, ua_values, ub_values, time_values):
#     """改进的符号变化检测算法（按时间步处理）"""
#     results = []
    
#     # 按x分组处理
#     for xvalue in np.unique(x_values):
#         xvalue_mask = (xvalue == x_values)
#         current_omega = omega_z[xvalue_mask]
#         current_y = y_values[xvalue_mask]
#         current_ua = ua_values[xvalue_mask]
#         current_ub = ub_values[xvalue_mask]
#         current_time = time_values[xvalue_mask]
        
#         # 符号变化检测
#         signs = np.sign(current_omega)
#         sign_changes = np.where(np.diff(signs, prepend=np.nan) != 0)[0]
        
#         # 处理每个符号变化点
#         valid_pairs = []
#         for i, idx in enumerate(sign_changes):
#             if idx == 0 or i == len(sign_changes)-1:
#                 continue
                
#             # ========== 第一阶段：寻找合格的负→正点 ==========
#             if current_y[idx] > 2e-3 and signs[idx-1] < 0 and signs[idx] > 0:
#                 # 检查后续5个点是否都为正（包含边界处理）
#                 end_pos = min(idx + 10, len(current_omega))
#                 next_five = current_omega[idx:end_pos]
                
#                 # 必须满足：1) 有至少5个点 2) 全部>=0
#                 if len(next_five) >= 5 and np.all(next_five >= 0):
#                     # ========== 第二阶段：寻找合格的正→负点 ==========
#                     for j, next_idx in enumerate(sign_changes[i+1:], start=i+1):
#                         if signs[next_idx-1] > 0 and signs[next_idx] < 0:
#                             # 检查后续5个点是否都为非正（包含边界处理）
#                             end_neg = min(next_idx + 10, len(current_omega))
#                             next_five_neg = current_omega[next_idx:end_neg]
                            
#                             # 必须满足：1) 有至少5个点 2) 全部<=0
#                             if len(next_five_neg) >= 5 and np.all(next_five_neg <= 0):
#                                 valid_pairs.append({
#                                     'time': current_time[idx],
#                                     'x_position': xvalue,
#                                     'y_start': current_y[idx],
#                                     'y_end': current_y[next_idx],
#                                     'delta_y': current_y[next_idx] - current_y[idx],
#                                     'omega_start': current_omega[idx],
#                                     'omega_end': current_omega[next_idx],
#                                     'u.a_start': current_ua[idx],
#                                     'u.a_end': current_ua[next_idx],
#                                     'u.b_start': current_ub[idx],
#                                     'u.b_end': current_ub[next_idx],
#                                     'delta_ua': current_ua[idx] - current_ua[next_idx],
#                                     'delta_ub': current_ub[idx] - current_ub[next_idx]
#                                 })
#                                 break  # 找到第一个符合条件的就退出
#                     break  # 每个负→正点只匹配第一个合格的正→负点
    
#         results.extend(valid_pairs)
    
#     return results

def detect_sign_changes(omega_z, y_values, x_values, ua_values, ub_values, time_values):
    """改进的符号变化检测算法（优先使用第二对符号变化点）"""
    results = []
    
    # 按x分组处理
    for xvalue in np.unique(x_values):
        xvalue_mask = (xvalue == x_values)
        current_omega = omega_z[xvalue_mask]
        current_y = y_values[xvalue_mask]
        current_ua = ua_values[xvalue_mask]
        current_ub = ub_values[xvalue_mask]
        current_time = time_values[xvalue_mask]
        
        # 符号变化检测
        signs = np.sign(current_omega)
        sign_changes = np.where(np.diff(signs, prepend=np.nan) != 0)[0]
        
        # 存储找到的对
        pairs_found = []
        
        for i, idx in enumerate(sign_changes):
            if idx == 0 or i == len(sign_changes)-1:
                continue
                
            # ========== 寻找合格的负→正点 ==========
            if current_y[idx] > 2e-3 and signs[idx-1] < 0 and signs[idx] > 0:
                # 检查后续5个点是否都为正
                end_pos = min(idx + 5, len(current_omega))
                next_five = current_omega[idx:end_pos]
                
                if len(next_five) >= 5 :
                    # ========== 寻找匹配的正→负点 ==========
                    for j, next_idx in enumerate(sign_changes[i+1:], start=i+1):
                        if signs[next_idx-1] > 0 and signs[next_idx] < 0 and current_y[next_idx] < 0.23:
                            # 检查后续5个点是否为非正
                            end_neg = min(next_idx + 5, len(current_omega))
                            next_five_neg = current_omega[next_idx:end_neg]
                            
                            if len(next_five_neg) >= 5: #and np.all(next_five_neg <= 0):
                                delta_y = current_y[next_idx] - current_y[idx]
                                if delta_y > 0.02:  # 确保y值是递增的
                                    pairs_found.append({
                                    'time': current_time[idx],
                                    'x_position': xvalue,
                                    'y_start': current_y[idx],
                                    'y_end': current_y[next_idx],
                                    'delta_y': current_y[next_idx] - current_y[idx],
                                    'omega_start': current_omega[idx],
                                    'omega_end': current_omega[next_idx],
                                    'u.a_start': current_ua[idx],
                                    'u.a_end': current_ua[next_idx],
                                    'u.b_start': current_ub[idx],
                                    'u.b_end': current_ub[next_idx],
                                    'delta_ua': current_ua[idx] - current_ua[next_idx],
                                    'delta_ub': current_ub[idx] - current_ub[next_idx],
                                    'pair_index': len(pairs_found) + 1  # 记录这是第几对
                                })
                                break  # 找到匹配的正→负点就退出内层循环
        
        # 决定使用哪一对
        if pairs_found:
            results.append(pairs_found[-1])  # 使用最后找到的一对
        # if len(pairs_found) == 2:
        #     results.append(pairs_found[1])  # 使用第二对
        # elif len(pairs_found) > 2:
        #     results.append(pairs_found[2])
        # elif len(pairs_found) == 1:
        #     results.append(pairs_found[0])  # 只有一对时使用第一对
    
    return results


def process_single_file(input_file, output_file):
    """处理单个文件并直接保存结果"""
    try:
        # 读取数据
        df = pd.read_csv(input_file)
        
        # 计算涡量
        df['omegaz'] = calculate_omegaz(df)
        
        # 提取二维切片 (z=0)
        slice_df = df[df['Points:2'] == 0]

        # selecting front position
        front_position = get_front_position(slice_df)
        
        # 执行符号变化检测
        if len(slice_df) < 3:
            print(f"跳过数据不足的文件: {input_file}")
            return False
            
        results = detect_sign_changes(
            omega_z=slice_df['omegaz'].values,
            y_values=slice_df['Points:1'].values,
            time_values=slice_df['Time'].values,
            x_values=slice_df['Points:0'].values,
            ua_values=slice_df['U.a:0'].values,
            ub_values=slice_df['U.b:0'].values
        )
        # 将front_position加入每条结果
        for r in results:
            r['front_position'] = front_position 

        # 保存结果
        if results:
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"成功保存 {len(results)} 条记录到 {output_file}")
        else:
            print(f"未检测到有效符号变化 {input_file}")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def main():
    # 准备文件列表
    input_files = sorted(RAW_DATA_DIR.glob(f"{FILE_PREFIX}_50.csv"))
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理每个文件
    success_count = 0
    for input_file in input_files:
        output_file = OUTPUT_DIR / get_output_filename(input_file)
        print(f"\n处理文件: {input_file.name} → {output_file.name}")
        
        if process_single_file(input_file, output_file):
            success_count += 1
    
    print(f"\n处理完成: 成功 {success_count}/{len(input_files)} 个文件")

if __name__ == "__main__":
    main()