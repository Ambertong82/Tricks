import pandas as pd
import numpy as np
from fractions import Fraction

# 常量定义
A = 1/4  # 修改这里即可自动更新文件名（支持分数/浮点数）
BASE_PATH = '/home/amber/postpro/'
FILE_PREFIX = 'case090429_5'

# 动态生成A的数字标识（处理分数和浮点数）
def get_a_identifier(a_value):
    # 将输入转换为分数形式（例如0.25→1/4）
    frac = Fraction(a_value).limit_denominator()
    return f"{frac.numerator}{frac.denominator}"  # 拼接分子分母

# 动态生成输出文件名
def get_output_files():
    a_id = get_a_identifier(A)  # 例如A=10/12→"1012"
    return {
        'yy': f'x{a_id}xyy.csv',
        'ua': f'x{a_id}xua.csv',
        'ub': f'x{a_id}xub.csv',
        'Rig': f'x{a_id}xRig.csv',
        'Rigg': f'x{a_id}xRigg.csv',
        'omegaz': f'x{a_id}xomegaz.csv',
        'reynolds': f'x{a_id}xReynolds[040].csv',
        'gradUb': f'x{a_id}xdudy.csv',
        'yplus': f'x{a_id}xYPLUS.csv',
        'alpha': f'x{a_id}xALPHA.csv'
    }

def calculate_derived_values(df, x, yvalue, c_value, grad_Ub, grad_dvdx, grad_dudy, nutb):
    """计算衍生量"""
    rho_mix = c_value * 2217 + 1000
    drhody = np.gradient(rho_mix, yvalue)
    dalphady = np.gradient(c_value, yvalue)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Rig = -9.81 * drhody / (1000 * grad_Ub**2)
        Rigg = dalphady / (grad_Ub**2)
        Rig = np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0)
        Rigg = np.nan_to_num(Rigg, nan=0.0, posinf=0.0, neginf=0.0)
    
    omegaz = grad_dvdx - grad_dudy
    reynolds = nutb * (grad_dudy + grad_dvdx)
    center = (yvalue[1:] - yvalue[:-1]) / 2 + yvalue[:-1]
    yplus = np.sqrt(1e-6 * grad_dudy[0]) * center / 1e-6
    
    return {
        'Rig': Rig.tolist(),
        'Rigg': Rigg.tolist(),
        'omegaz': omegaz.tolist(),
        'reynolds': reynolds.tolist(),
        'yplus': yplus.tolist(),
        'gradUb': grad_Ub.tolist(),
        'alpha': c_value.tolist()
    }

def process_file(file):
    """处理单个文件"""
    df = pd.read_csv(file)
    filtered_df = df[(df['alpha.a'] > 0.00001) & (df['Points:1'] > 0)]
    
    if len(filtered_df) <= 1:
        return None
    
    max_point = filtered_df['Points:0'].idxmax()
    time = filtered_df.loc[max_point, 'Time']
    x = filtered_df.loc[max_point, 'Points:0'] - A * 0.3
    closest_point = (filtered_df['Points:0'] - x).abs().idxmin()
    x = filtered_df.loc[closest_point, 'Points:0']
    
    # 提取基础数据
    mask = (df['Points:0'] == x) & (df['Points:2'] == 0)
    yvalue = df.loc[mask, 'Points:1'].values
    uavalue = df.loc[mask, 'U.a:0'].values
    ubvalue = df.loc[mask, 'U.b:0'].values
    c_value = df.loc[mask, 'alpha.a'].values
    grad_Ub = df.loc[mask, 'grad(U.b):3'].values
    grad_dvdx = df.loc[mask, 'grad(U.b):1'].values
    grad_dudy = df.loc[mask, 'grad(U.b):3'].values
    nutb = df.loc[mask, 'nut.b'].values
    
    # 计算衍生量
    derived = calculate_derived_values(df, x, yvalue, c_value, grad_Ub, grad_dvdx, grad_dudy, nutb)
    
    return {
        'time': time,
        'x': x,
        'yy': yvalue.tolist(),
        'ua': uavalue.tolist(),
        'ub': ubvalue.tolist(),
        **derived
    }

def save_data(data_dict):
    """保存数据到CSV文件"""
    OUTPUT_FILES = get_output_files()  # 动态获取文件名
    for key, filename in OUTPUT_FILES.items():
        data = []
        for item in data_dict:
            data.append([item['time'], item['x']] + item.get(key, []))
        
        df = pd.DataFrame(data).transpose()
        df.to_csv(f"{BASE_PATH}{FILE_PREFIX}_{filename}", index=False, header=False)

def main():
    file_list = [f'/home/amber/postpro/rawdata/{FILE_PREFIX}_{i}.csv' for i in range(0, 78)]
    results = []
    
    for file in file_list:
        result = process_file(file)
        if result:
            results.append(result)
    
    if results:
        save_data(results)
        print(f"数据处理完成(A={A})，结果已保存到以下文件：")
        for name in get_output_files().values():
            print(f"  - {FILE_PREFIX}_{name}")
    else:
        print("未找到有效数据")

if __name__ == '__main__':
    main()