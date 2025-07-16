import pandas as pd
import numpy as np
from fractions import Fraction

# 常量定义
A = 1/2  # 修改这里即可自动更新文件名（支持分数/浮点数）
B = 1e-5
BASE_PATH = '/home/amber/postpro/'
FILE_PREFIX = 'case230427_4'  # 修改这里即可自动更新文件名

# 动态生成A的数字标识（处理分数和浮点数）


def get_a_identifier(a_value):
    # 将输入转换为分数形式（例如0.25→1/4）
    frac = Fraction(a_value).limit_denominator()
    return f"{frac.numerator}{frac.denominator}"  # 拼接分子分母

# 动态生成输出文件名


def get_output_files():
    a_id = get_a_identifier(A) # 例如A=10/12→"1012"
    return {
        'yy': f'x{a_id}xyy.csv',
        'ua': f'x{a_id}xua.csv',
        'ub': f'x{a_id}xub.csv',
        'Rig': f'x{a_id}xRig.csv',
        'Rigg': f'x{a_id}xRigg.csv',
        'omegaz': f'x{a_id}xomegaz.csv',
        'reynolds12': f'x{a_id}xReynolds.csv',
        'reynolds11': f'x{a_id}xReynolds11.csv',
        'reynolds22': f'x{a_id}xReynolds22.csv',
        'yplus': f'x{a_id}xYPLUS.csv',
        'alpha': f'x{a_id}xALPHA.csv',
        'uadimless': f'x{a_id}xuadimless.csv',
        'ubdimless': f'x{a_id}xubdimless.csv',
        'shearstress': f'x{a_id}xShearStress.csv',
        'kinetic_energy': f'x{a_id}xKineticEnergy.csv',
        'production': f'x{a_id}xProduction.csv',
        'production_dudx': f'x{a_id}xProduction_dudx.csv',
        'production_dvdy': f'x{a_id}xProduction_dvdy.csv',
        'grad_dudx': f'x{a_id}xGrad_dudx.csv',
        'grad_dvdy': f'x{a_id}xGrad_dvdy.csv',
        'dragpart': f'x{a_id}xDragPart.csv',
        'buoyancy': f'x{a_id}xBuoyancy.csv',
        'dissipation': f'x{a_id}xDissipation.csv',
        'uay': f'x{a_id}xuay.csv',
        'uby': f'x{a_id}xuby.csv',
        'ke_dimless': f'x{a_id}xke_dimless.csv',
    }


def calculate_derived_values(
        df,
        x,
        yvalue,
        alpha,
        kinetic_energy,
        gamma,
        grad_dudx,
        grad_dvdx,
        grad_dudy,
        grad_dvdy,
        nutb,
        uavalue,
        ubvalue,
        uavalueyy,
        ubvalueyy,
        grad_alpha1,
        grad_alpha2,
        grad_beta1,
        grad_beta2,
        omega):
    """计算衍生量"""
    rho_mix = alpha * 2217 + 1000
    drhody = np.gradient(rho_mix, yvalue)
    dalphady = np.gradient(alpha, yvalue)

    with np.errstate(divide='ignore', invalid='ignore'):
        Rig = -9.81 * drhody / (1000 * grad_dudy**2)
        Rigg =  -9.81*2217*grad_alpha2 / (1000 * grad_dudy**2)
        Rig = np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0)
        Rigg = np.nan_to_num(Rigg, nan=0.0, posinf=0.0, neginf=0.0)

    omegaz = grad_dvdx - grad_dudy
    reynolds12 = nutb * (grad_dudy + grad_dvdx)*1000
    reynolds11 = nutb * (grad_dudx + grad_dvdx-2/3*kinetic_energy)*1000
    reynolds22 = nutb * (grad_dvdy + grad_dvdy-2/3*kinetic_energy)*1000
    center = (yvalue[1:] - yvalue[:-1]) / 2 + yvalue[:-1]
    yplus = np.sqrt(1e-6 * grad_dudy[0]) * center / 1e-6
    shearstress = grad_dudy+grad_dvdx
    production = alpha*1000*nutb * (2*grad_dudx**2 + 2*grad_dvdy**2+(grad_dvdx+grad_dudy)**2) 
    production_dudx = alpha*1000*nutb * (2*grad_dudx**2)- alpha*2/3*1000*kinetic_energy *grad_dudx
    production_dvdy = alpha*1000*nutb * (2*grad_dvdy**2)- alpha*2/3*1000*kinetic_energy *grad_dvdy
    dragpart = gamma * ((ubvalue-uavalue)*grad_alpha1 + (ubvalueyy-uavalueyy)*grad_alpha2)* nutb /(1-alpha)
    tauxx = 1e-3*grad_dudx*2
    tauxy = 1e-3*(grad_dudy+grad_dvdx)
    tauyy = 1e-3*grad_dvdy*2
    seoxx = grad_beta1**2/(1-alpha)
    seoxy = (grad_beta1*grad_beta2)/(1-alpha)
    seoyy = grad_beta2**2/(1-alpha)
    buoyancy = nutb*(tauxx*seoxx + 2*tauxy*seoxy + tauyy*seoyy)
    dissipation = -(1-alpha)*1000*0.09*kinetic_energy*omega

    return {
        'Rig': Rig.tolist(),
        'Rigg': Rigg.tolist(),
        'omegaz': omegaz.tolist(),
        'reynolds12': reynolds12.tolist(),
        'reynolds11': reynolds11.tolist(),
        'reynolds22': reynolds22.tolist(),
        'yplus': yplus.tolist(),
        'grad_dudx': grad_dudx.tolist(),
        'grad_dvdy': grad_dvdy.tolist(),
        'alpha': alpha.tolist(),
        'shearstress': shearstress.tolist(),
        'production': production.tolist(),
        'production_dudx': production_dudx.tolist(),
        'production_dvdy': production_dvdy.tolist(),
        'dragpart': dragpart.tolist(),
        'buoyancy': buoyancy.tolist(),
        'dissipation': dissipation.tolist(),
    }


def process_file(file):
    """处理单个文件"""
    df = pd.read_csv(file)
    filtered_df = df[(df['alpha.a'] > 1e-5) & (df['Points:1'] > 0)]

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
    uavalueyy = df.loc[mask, 'U.a:1'].values
    ubvalueyy = df.loc[mask, 'U.b:1'].values
    uavaluedimless = uavalue / 0.27
    ubvaluedimless = ubvalue / 0.27
    alpha = df.loc[mask, 'alpha.a'].values
    #grad_Ub = df.loc[mask, 'grad(U.b):3'].values
    grad_dudx = df.loc[mask, 'grad(U.b):0'].values
    grad_dvdx = df.loc[mask, 'grad(U.b):1'].values
    grad_dudy = df.loc[mask, 'grad(U.b):3'].values
    grad_dvdy = df.loc[mask, 'grad(U.b):4'].values
    nutb = df.loc[mask, 'nut.b'].values
    kinetic_energy = df.loc[mask, 'k.b'].values
    max_ke = kinetic_energy.max()
    ke_dimless = kinetic_energy / max_ke if max_ke != 0 else kinetic_energy
    gamma = df.loc[mask, 'K'].values
    grad_alpha1 = df.loc[mask, 'grad(alpha.a):0'].values
    grad_alpha2 = df.loc[mask, 'grad(alpha.a):1'].values
    grad_beta1 = df.loc[mask, 'grad(alpha.b):0'].values
    grad_beta2 = df.loc[mask, 'grad(alpha.b):1'].values
    omega = df.loc[mask, 'omega.b'].values
    

    # 计算衍生量
    derived = calculate_derived_values(
        df, x, yvalue, alpha,  kinetic_energy, gamma, grad_dudx, grad_dvdx, grad_dudy, grad_dvdy,nutb, 
        uavalue, ubvalue,uavalueyy, ubvalueyy, grad_alpha1, grad_alpha2, grad_beta1, grad_beta2,omega)

    return {
        'time': time,
        'timedimless': time / 0.56,  # 假设0.28是时间的基准值
        'x': x,
        'yy': yvalue.tolist(),
        'ua': uavalue.tolist(),
        'ub': ubvalue.tolist(),
        'uadimless': uavaluedimless.tolist(),
        'ubdimless': ubvaluedimless.tolist(),
        'kinetic_energy': kinetic_energy.tolist(),
        'grad_dvdy': grad_dvdy.tolist(),
        'grad_dudx': grad_dudx.tolist(),
        'uay': uavalueyy.tolist(),
        'uby': ubvalueyy.tolist(),
        'ke_dimless': ke_dimless.tolist(),
        **derived
    }



def save_data(data_dict):
    """保存数据到CSV文件（带自动匹配timedimless和转置）"""
    OUTPUT_FILES = get_output_files()
    
    for key, filename in OUTPUT_FILES.items():
        data = []
        
        # 判断是否使用timedimless
        use_timedimless = any(k in key for k in ['uadimless', 'ubdimless'])
        
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
            f"{BASE_PATH}{FILE_PREFIX}_{filename}",
            index=False,
            header=False
        )


def main():
    file_list = [
        f'/home/amber/postpro/rawdata/{FILE_PREFIX}_{i}.csv' for i in range(1, 79, 2)]
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
