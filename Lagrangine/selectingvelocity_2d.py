import numpy as np
import pandas as pd
import os
import fluidfoam
from fractions import Fraction
### this code is used to extract the data at specific x location (head-0.3) at every time step #####

# 常量定义
A = 1/3  # 修改这里即可自动更新文件名（支持分数/浮点数）
y_min = 0
alpha_threshold = 1e-5



def get_a_identifier(a_value):
    frac = Fraction(a_value).limit_denominator()
    return f"{frac.numerator}{frac.denominator}"

def get_output_files():
    a_id = get_a_identifier(A) # 例如A=10/12→"1012"
    return {
        'yy': f'x{a_id}xyy.csv',
        'ua': f'x{a_id}xua.csv',
        'ub': f'x{a_id}xub.csv',
        # 'Rig': f'x{a_id}xRig.csv',
        # 'Rigg': f'x{a_id}xRigg.csv',
        # 'omegaz': f'x{a_id}xomegaz.csv',
        # 'reynolds12': f'x{a_id}xReynolds.csv',
        # 'reynolds11': f'x{a_id}xReynolds11.csv',
        # 'reynolds22': f'x{a_id}xReynolds22.csv',
        'yplus': f'x{a_id}xYPLUS.csv',
        'G': f'x{a_id}xG.csv',
        
        # 'alpha': f'x{a_id}xALPHA.csv',
        # 'uadimless': f'x{a_id}xuadimless.csv',
        # 'ubdimless': f'x{a_id}xubdimless.csv',
        # 'shearstress': f'x{a_id}xShearStress.csv',
        # 'kinetic_energy': f'x{a_id}xKineticEnergy.csv',
        # 'production': f'x{a_id}xProduction.csv',
        # 'production_xy': f'x{a_id}xProduction_xy.csv',
        # 'production_dudx': f'x{a_id}xProduction_dudx.csv',
        # 'production_dvdy': f'x{a_id}xProduction_dvdy.csv',
        # 'grad_dudx': f'x{a_id}xGrad_dudx.csv',
        # 'grad_dvdy': f'x{a_id}xGrad_dvdy.csv',
        # 'dragpart': f'x{a_id}xDragPart.csv',
        # 'buoyancy': f'x{a_id}xBuoyancy.csv',
        # 'dissipation': f'x{a_id}xDissipation.csv',
        # 'uay': f'x{a_id}xuay.csv',
        # 'uby': f'x{a_id}xuby.csv',
        # 'ke_dimless': f'x{a_id}xke_dimless.csv',
        'H': f'x{a_id}xH.csv',
        'U': f'x{a_id}xU.csv',
        'ALPHA': f'x{a_id}xALPHA.csv',
        'H_depth': f'x{a_id}xH_depth.csv',
        # 'H_alpha': f'x{a_id}xH_alpha.csv',
        # 'U_alpha': f'x{a_id}xU_alpha.csv',
        # 'ALPHA_alpha': f'x{a_id}xALPHA_alpha.csv',
        # 'H_depth_alpha': f'x{a_id}xH_depth_alpha.csv',
        # 'advection': f'x{a_id}xAdvection.csv',
        # 'vorticity': f'x{a_id}xVorticity.csv',
        # 'ycrossing': f'x{a_id}xYcrossing.csv',
        # 'uhat': f'x{a_id}xu_hat.csv',
        # 'dragpart1': f'x{a_id}xDragPart1.csv',
        # 'dragpart2': f'x{a_id}xDragPart2.csv',
        # 'drag1': f'x{a_id}xDrag1.csv',
        # 'transport': f'x{a_id}xTransport.csv',
        # 'FX': f'x{a_id}xFX.csv',    
        # 'FY': f'x{a_id}xFY.csv',
        # 'k.b': f'x{a_id}xk_b.csv',
    }




def process_time_step(sol, time_v, X, Y,dx,dy):
    """处理单个时间步"""
    # 读取场数据
    
    Ua = fluidfoam.readvector(sol, str(time_v), "U.a")
    Ub = fluidfoam.readvector(sol, str(time_v), "U.b")
    alpha = fluidfoam.readscalar(sol, str(time_v), "alpha.a")
    nutb = fluidfoam.readscalar(sol, str(time_v), "nut.b")
    kb = fluidfoam.readscalar(sol, str(time_v), "k.b")
    omegab = fluidfoam.readscalar(sol, str(time_v), "omega.b")
    gradUb = fluidfoam.readtensor(sol, str(time_v), "grad(U.b)")
   
    gamma = fluidfoam.readscalar(sol, str(time_v), "K")

    Gpre = 2*(gradUb[0]**2 + gradUb[4]**2 + gradUb[8]**2) + \
        (gradUb[1]+gradUb[3])**2 + (gradUb[2]+gradUb[6])**2 + (gradUb[5]+gradUb[7])**2 
    G = Gpre*(1-alpha)*1000*nutb
    

    # 定位头部位置
    head_x = None
    for x in np.unique(X):
        mask = (X == x) & (Y >= y_min) & (alpha > alpha_threshold)
        if np.any(mask):
            head_x = x
    if head_x is None:
        print(f"Warning: No head found at t={time_v}")
        return None

    # 找到最近的网格点
    target_x = head_x - A * 0.3
    unique_x = np.unique(X)
    closest_x = unique_x[np.argmin(np.abs(unique_x - target_x))]
    
    # # 提取数据
    # mask = np.isclose(X, closest_x, atol=1e-6, rtol=1e-6)
    # if not np.any(mask):
    #     print(f"Warning: No points found at x={closest_x:.3f}m for t={time_v}")
    #     return None
    
    mask = (X == closest_x) & (Y >= 0)

    yvalue = Y[mask]
    xvalue = X[mask]
    uavalue = Ua[0][mask]
    uavalueorigin = Ua[0][mask]
    
    ubvalue = Ub[0][mask]
    uavalueyy = Ua[1][mask]
    ubvalueyy = Ub[1][mask]
    alpha_value = alpha[mask]
    nutb_value = nutb[mask]
    kinetic_energy = kb[mask]
    max_ke = kinetic_energy.max()
    ke_dimless = kinetic_energy / max_ke if max_ke != 0 else kinetic_energy
    omega_value = omegab[mask]
    G_value = G[mask]

    dx = dx[mask]
    dy = dy[mask]



    return {
        'time': time_v,
        'timedimless': time_v / 0.56,
        'x': closest_x,
        'yy': yvalue.tolist(),
        'ua': uavalueorigin.tolist(),
        'ub': ubvalue.tolist(),
        'uadimless': (uavalue / 0.27).tolist(),
        'ubdimless': (ubvalue / 0.27).tolist(),
        'kinetic_energy': kinetic_energy.tolist(),

        'uay': uavalueyy.tolist(),
        'uby': ubvalueyy.tolist(),
        'ke_dimless': ke_dimless.tolist(),
        'k.b': kb.tolist(),
        'G': G_value.tolist(),
    }

def save_data(data_dict, BASE_PATH, FILE_PREFIX):
    """保存数据到CSV文件（与原函数相同）"""
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
           # row = [time_col, item['x']] + [item.get(key.replace('timedimless_', ''), [])]

            data.append(row)
        
        # 转换为DataFrame并转置
        df = pd.DataFrame(data).transpose()
        
        # 保存文件（无表头）
        df.to_csv(
            f"{BASE_PATH}{FILE_PREFIX}_{filename}",
            index=False,
            header=False
        )
    # ... (保持与原函数相同的实现)

def main():
    # sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
    #sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4fine"

    sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/NEW/case230304_1"

    X, Y, Z = fluidfoam.readmesh(sol)
    dx = np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=0)
    times = np.arange(3, 15, 1)  # 对应原来的1-79,步长2
    results = []
    BASE_PATH = '/home/amber/postpro/selecting_variant/'
    # FILE_PREFIX = 'case230427_4'  # 修改这里即可自动更新文件名
    FILE_PREFIX = 'case230304_1'  # 修改这里即可自动更新文件名
    #FILE_PREFIX = 'case530628_1'  # 修改这里即可自动更新文件名
    # FILE_PREFIX = 'case231020_6'  # 修改这里即可自动更新文件名

    for time_v in times:
        result = process_time_step(sol, time_v, X, Y,dx,dy)
        if result:
            results.append(result)

    if results:
        save_data(results,BASE_PATH,FILE_PREFIX)
        print(f"数据处理完成(A={A})，结果已保存到以下文件：")
        for name in get_output_files().values():
            print(f"  - {FILE_PREFIX}_{name}")
    else:
        print("未找到有效数据")

if __name__ == '__main__':
    main()
