import pandas as pd
import numpy as np
from fractions import Fraction

# 常量定义
A = 1 / 4  # 修改这里即可自动更新文件名（支持分数/浮点数）
H = 0.3
ts = 1.86
ub = 0.08
BASE_PATH = '/home/amber/postpro/'
FILE_PREFIX = 'case0704_4'  # 修改这里即可自动更新文件名

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
        'uadimless': f'x{a_id}xuadimless.csv',
        'reynolds11': f'x{a_id}xReynolds11.csv',
        'reynolds12': f'x{a_id}xReynolds12.csv',
        'reynolds22': f'x{a_id}xReynolds22.csv',
        'gradUb': f'x{a_id}xdudy.csv',
        'alpha': f'x{a_id}xALPHA.csv',
        'shearstress': f'x{a_id}xShearStress.csv',
        'omegaz': f'x{a_id}xomegaz.csv',
        'k': f'x{a_id}xk.csv',
        'Rig': f'x{a_id}xRig.csv',
        'production': f'x{a_id}xProduction.csv',
        'dissipation': f'x{a_id}xDissipation.csv',
        'omega': f'x{a_id}xomega.csv',

    }


def calculate_derived_values(
        df,
        x,
        yvalue,
        alpha,
        grad_dudx,
        grad_dvdy,
        grad_dvdx,
        grad_dudy,
        kinetic_energy,
        u,
        omega,
        grad_alpha1,
        grad_alpha2,
        nutb,
        alpha_dy):
    """计算衍生量"""
    rho_mix = alpha * 200 + 1000
    drhody = np.gradient(rho_mix, yvalue)
    dalphady = np.gradient(alpha, yvalue)

    with np.errstate(divide='ignore', invalid='ignore'):
        Rig = -9.81 * alpha_dy * 2217 / (1000 * grad_dudy**2)
        Rigg = dalphady / (grad_dudy**2)
        Rig = np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0)
        Rigg = np.nan_to_num(Rigg, nan=0.0, posinf=0.0, neginf=0.0)

    omegaz = grad_dvdx - grad_dudy
    reynolds12 = nutb * (grad_dudy + grad_dvdx)*1000
    reynolds11 = nutb * (grad_dudx + grad_dvdx-2/3*kinetic_energy)*1000
    reynolds22 = nutb * (grad_dvdy + grad_dvdy-2/3*kinetic_energy)*1000
    center = (yvalue[1:] - yvalue[:-1]) / 2 + yvalue[:-1]
    yplus = np.sqrt(1e-6 * grad_dudy[0]) * center / 1e-6
    shearstress = grad_dudy+grad_dvdx
    production = (2217*alpha+1000)*nutb * (2*grad_dudx**2 + 2*grad_dvdy**2+(grad_dvdx+grad_dudy)**2) 
    production_dudx = alpha*1000*nutb * (2*grad_dudx**2)
    production_dvdy = alpha*1000*nutb * (2*grad_dvdy**2)-alpha*2/3*1000*kinetic_energy *grad_dvdy
    tauxx = 1e-3*grad_dudx*2
    tauxy = 1e-3*(grad_dudy+grad_dvdx)
    tauyy = 1e-3*grad_dvdy*2
    seoxx = grad_alpha1**2/(1-alpha)
    seoxy = (grad_alpha1*grad_alpha2)/(1-alpha)
    seoyy = grad_alpha2**2/(1-alpha)
    buoyancy = nutb*(tauxx*seoxx + 2*tauxy*seoxy + tauyy*seoyy)
    dissipation = -(2217*alpha+1000)*0.09*kinetic_energy*omega

    return {
        'Rig': Rig.tolist(),
        'Rigg': Rigg.tolist(),
        'omegaz': omegaz.tolist(),
        'reynolds12': reynolds12.tolist(),
        'reynolds11': reynolds11.tolist(),
        'reynolds22': reynolds22.tolist(),
        'production': production.tolist(),
        'shearstress': shearstress.tolist(),
        'yplus': yplus.tolist(),
        'gradUb': grad_dudy.tolist(),
        'alpha': alpha.tolist(),
        'dissipation': dissipation.tolist(),
    }



def process_file(file):
    """处理单个文件"""
    df = pd.read_csv(file)
    filtered_df = df[(df['alpha.saline'] > 0.00001) & (df['Points:1'] > 0)]

    if len(filtered_df) <= 1:
        return None

    max_point = filtered_df['Points:0'].idxmax()
    time = filtered_df.loc[max_point, 'Time'] 
    x = filtered_df.loc[max_point, 'Points:0'] - A * H
    closest_point = (filtered_df['Points:0'] - x).abs().idxmin()
    x = filtered_df.loc[closest_point, 'Points:0']

    # 提取基础数据
    mask = (df['Points:0'] == x) & (df['Points:2'] == 0)
    yvalue = df.loc[mask, 'Points:1'].values
    u = df.loc[mask, 'U:0'].values
    uadimless = u / ub  # 无量纲化
    grad_dvdx = df.loc[mask, 'grad(U):1'].values
    grad_dudy = df.loc[mask, 'grad(U):3'].values
    alpha = df.loc[mask, 'alpha.saline'].values
    k_value = df.loc[mask, 'k'].values
    nutb = df.loc[mask, 'nut'].values
    alpha_dy = df.loc[mask, 'grad(alpha.saline):1'].values
    grad_dudx = df.loc[mask, 'grad(U):0'].values
    grad_dvdy = df.loc[mask, 'grad(U):2'].values
    kinetic_energy = df.loc[mask, 'k'].values
    omega = df.loc[mask, 'omega'].values
    grad_alpha1 = df.loc[mask, 'grad(alpha.saline):0'].values
    grad_alpha2 = df.loc[mask, 'grad(alpha.saline):1'].values

    # 计算衍生量
    derived = calculate_derived_values(
        df,
        x,
        yvalue,
        alpha,
        grad_dudx,
        grad_dvdy,
        grad_dvdx,
        grad_dudy,
        kinetic_energy,
        u,
        omega,
        grad_alpha1,
        grad_alpha2,
        nutb,
        alpha_dy)

    return {
        'time': time,
        'timedimless': time / ts,  # 假设0.28是时间的基准值
        'x': x,
        'uadimless': uadimless.tolist(),
        'yy': yvalue.tolist(),
        'ua': u.tolist(), 
        'k': k_value.tolist(), 
        'alphady': alpha_dy.tolist(),
        'omega':omega.tolist(),
        **derived
    }


def save_data(data_dict):
    """保存数据到CSV文件（带自动匹配timedimless和转置）"""
    OUTPUT_FILES = get_output_files()
    OUTPUT_DIR = '/home/amber/postpro/Marino/'
    
    for key, filename in OUTPUT_FILES.items():
        data = []
        
        # 判断是否使用timedimless
        use_timedimless = any(k in key for k in ['uadimless'])
        
        for item in data_dict:
            # 选择时间列（timedimless或原始time）
            time_col = item['timedimless'] if use_timedimless else item['time']
            
            # 构建数据行（时间 + x + 数据）
            row = [time_col, item['x']] + item.get(key.replace('timedimless_', ''), [])
            data.append(row)
        
        # 转换为DataFrame并转置
        df = pd.DataFrame(data).transpose()
        
        # 保存文件（无表头）
        df.to_csv(
            f"{OUTPUT_DIR}{FILE_PREFIX}_{filename}",
            index=False,
            header=False
        )


def main():
    file_list = [
        f'/home/amber/postpro/rawdata/{FILE_PREFIX}_{i}.csv' for i in range(0, 59)]
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
