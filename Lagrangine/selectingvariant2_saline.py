import numpy as np
import pandas as pd
import os
import fluidfoam
from fractions import Fraction
## this code is used to extract the data at specific x location (head-0.3) at every time step #####
# 常量定义
A = 1/1  # 修改这里即可自动更新文件名（支持分数/浮点数）
B = 1e-5
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
        'Rig': f'x{a_id}xRig.csv',
        'Rigg': f'x{a_id}xRigg.csv',
        'omegaz': f'x{a_id}xomegaz.csv',
        'reynolds12': f'x{a_id}xReynolds.csv',
        'reynolds11': f'x{a_id}xReynolds11.csv',
        'reynolds22': f'x{a_id}xReynolds22.csv',
        'yplus': f'x{a_id}xYPLUS.csv',
        'alpha': f'x{a_id}xALPHA.csv',
        # 'uadimless': f'x{a_id}xuadimless.csv',
        # 'ubdimless': f'x{a_id}xubdimless.csv',
        # 'shearstress': f'x{a_id}xShearStress.csv',
        # 'kinetic_energy': f'x{a_id}xKineticEnergy.csv',
        # 'production': f'x{a_id}xProduction.csv',
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
        'H_alpha': f'x{a_id}xH_alpha.csv',
        'U_alpha': f'x{a_id}xU_alpha.csv',
    }


def calculate_derived_values(yvalue, alpha, kinetic_energy,  grad_dudx, grad_dvdx, 
                           grad_dudy, grad_dvdy, nutb, uavalue, uavalueyy, 
                            grad_alpha1, grad_alpha2, omega):
    """计算衍生量"""
    rho_mix = alpha * 2217 + 1000
    drhody = np.gradient(rho_mix, yvalue)
    dalphady = np.gradient(alpha, yvalue)

    with np.errstate(divide='ignore', invalid='ignore'):
        Rig = -9.81 * drhody / (1000 * grad_dudy**2)
        Rigg = -9.81*2217*grad_alpha2 / (1000 * grad_dudy**2)
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
    #dragpart = gamma * ((ubvalue-uavalue)*grad_alpha1 + (ubvalueyy-uavalueyy)*grad_alpha2)* nutb /(1-alpha)
    tauxx = 1e-3*grad_dudx*2
    tauxy = 1e-3*(grad_dudy+grad_dvdx)
    tauyy = 1e-3*grad_dvdy*2

    dissipation = -(1-alpha)*1000*0.09*kinetic_energy*omega
        # 寻找速度正负交界点
    sign_changes = np.where(np.diff(np.sign(uavalue)))[0]
    max_ya_crossing_index = sign_changes[np.argmax(yvalue[sign_changes])] + 1 if len(sign_changes) > 0 else len(yvalue) - 1
    y_crossing = yvalue[max_ya_crossing_index]
    u_crossing = uavalue[max_ya_crossing_index]
        
    # Vectorized integration
    ua_alpha = uavalue * alpha
    integral = np.trapz(ua_alpha[:max_ya_crossing_index], yvalue[:max_ya_crossing_index])
    integralU = np.trapz(uavalue[:max_ya_crossing_index], yvalue[:max_ya_crossing_index])
    integralU2 = np.trapz(uavalue[:max_ya_crossing_index]**2, yvalue[:max_ya_crossing_index])
    integral2 = np.trapz((uavalue[:max_ya_crossing_index] * alpha[:max_ya_crossing_index])**2, yvalue[:max_ya_crossing_index])
        
    # Calculate derived quantities
    U = integralU2 / integralU if integralU != 0 else 0
    H = integral**2 / integral2 if integral2 != 0 else 0
    ALPHA = integral / integralU if integralU != 0 else 0
    H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0
    

        # Find the y-coordinate where alpha crosses below 1e-5
        # 首先筛选出 y > 0.005 的点
    y_threshold = 0.005
    valid_mask = yvalue > y_threshold

    if np.any(valid_mask):
            # 在有效范围内寻找 alpha < 1e-5 的点
            alpha_threshold = 1e-5
            below_threshold = (alpha[valid_mask] < alpha_threshold)
            
            if np.any(below_threshold):
                # 找到第一个满足条件的点的相对索引
                first_below_rel_index = np.argmax(below_threshold)
                # 转换为原始数组中的绝对索引
                valid_indices = np.where(valid_mask)[0]
                max_ya_crossing_index_alpha = valid_indices[first_below_rel_index]
                y_crossing_alpha = yvalue[max_ya_crossing_index_alpha]  # 修正变量名
                u_crossing_alpha = uavalue[max_ya_crossing_index_alpha]
                #print(f"Found y_crossing at {y_crossing_alpha} for xx={xx}")
            else:
                # 如果没有找到，使用有效范围内的最大y值
                max_ya_crossing_index_alpha = np.where(valid_mask)[0][-1]
                y_crossing_alpha = yvalue[max_ya_crossing_index_alpha]  # 修正变量名
                u_crossing_alpha = uavalue[max_ya_crossing_index_alpha]
                #print(f"No alpha < {alpha_threshold} found above y={y_threshold} for xx={xx}, using max y={y_crossing_alpha}")
    else:
            # 如果没有y>0.005的点，使用最后一个点
            max_ya_crossing_index_alpha = len(yvalue) - 1
            y_crossing_alpha = yvalue[max_ya_crossing_index_alpha]  # 修正变量名
            u_crossing_alpha = uavalue[max_ya_crossing_index_alpha]  # 无法定义

        # Vectorized integration
    ua_alpha_alpha = uavalue * alpha
    integral_alpha = np.trapz(ua_alpha_alpha[:max_ya_crossing_index_alpha], yvalue[:max_ya_crossing_index_alpha])
    integralU_alpha = np.trapz(uavalue[:max_ya_crossing_index_alpha], yvalue[:max_ya_crossing_index_alpha])
    integralU2_alpha = np.trapz(uavalue[:max_ya_crossing_index_alpha]**2, yvalue[:max_ya_crossing_index_alpha])
    integral2_alpha = np.trapz((uavalue[:max_ya_crossing_index_alpha] * alpha[:max_ya_crossing_index_alpha])**2, yvalue[:max_ya_crossing_index_alpha])
        
        # Calculate derived quantities
    U_alpha = integralU2_alpha / integralU_alpha if integralU_alpha != 0 else 0
    H_alpha = integral_alpha**2 / integral2_alpha if integral2_alpha != 0 else 0
    ALPHA_alpha = integral_alpha / integralU_alpha if integralU_alpha != 0 else 0
    H_depth_alpha = integralU_alpha**2 / integralU2_alpha if integralU2_alpha != 0 else 0

        
        # # Depth-averaged quantities
        # p_k_average_alpha = np.trapz(P_k[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0
        # epsilon_average_alpha = np.trapz(epsilon_alpharho[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0
        # G_average_alpha = np.trapz(G[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0
        # G2_average_alpha = np.trapz(G2[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0



    #     # Dimensionless numbers
    # denominator = g * R * ALPHA * H
    # Fr = U / np.sqrt(denominator) if denominator > 0 else np.nan
    # Re = U * H / nu
        
    #     # Drag coefficient
    # grad_Ub0 = grad_dudy[0] if len(grad_dudy) > 0 else 0
    # u_star2 = grad_Ub0 * nu
    # Cd = u_star2 / (U**2) if U != 0 else np.nan
    #     #print(np.max(x_coords))


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
        #'dragpart': dragpart.tolist(),
        #'buoyancy': buoyancy.tolist(),
        'dissipation': dissipation.tolist(),
        'H': [H],
        'U': [U],
        'ALPHA': [ALPHA],
        'H_depth': [H_depth],
        'H_alpha': [H_alpha],
        'U_alpha': [U_alpha],
    
    }

def process_time_step(sol, time_v, X, Y):
    """处理单个时间步"""
    # 读取场数据
    
    Ua = fluidfoam.readvector(sol, str(time_v), "U")
    
    alpha = fluidfoam.readscalar(sol, str(time_v), "alpha.saline")
    nutb = fluidfoam.readscalar(sol, str(time_v), "nut")
    kb = fluidfoam.readscalar(sol, str(time_v), "k")
    omegab = fluidfoam.readscalar(sol, str(time_v), "omega")
    grad_Ub = fluidfoam.readtensor(sol, str(time_v), "grad(U)")
    grad_alpha = fluidfoam.readvector(sol, str(time_v), "grad(alpha.saline)")
    
    #gamma = fluidfoam.readscalar(sol, str(time_v), "K")

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
    
    # 提取数据
    mask = np.isclose(X, closest_x, atol=1e-6, rtol=1e-6)
    if not np.any(mask):
        print(f"Warning: No points found at x={closest_x:.3f}m for t={time_v}")
        return None

    yvalue = Y[mask]
    uavalue = Ua[0][mask]
    
    uavalueyy = Ua[1][mask]
    
    alpha_value = alpha[mask]
    nutb_value = nutb[mask]
    kinetic_energy = kb[mask]
    max_ke = kinetic_energy.max()
    ke_dimless = kinetic_energy / max_ke if max_ke != 0 else kinetic_energy
    omega_value = omegab[mask]
    grad_dudx = grad_Ub[0][mask]
    grad_dudy = grad_Ub[3][mask]
    grad_dvdx = grad_Ub[1][mask]
    grad_dvdy = grad_Ub[4][mask]
    grad_alpha1 = grad_alpha[0][mask]
    grad_alpha2 = grad_alpha[1][mask]
    

    # 计算衍生量
    derived = calculate_derived_values(
        yvalue, alpha_value, kinetic_energy, grad_dudx, grad_dvdx,
        grad_dudy, grad_dvdy, nutb_value, uavalue,  uavalueyy, 
        grad_alpha1, grad_alpha2,  omega_value
    )

    return {
        'time': time_v,
        'timedimless': time_v / 0.56,
        'x': closest_x,
        'yy': yvalue.tolist(),
        'ua': uavalue.tolist(),
        'uadimless': (uavalue / 0.27).tolist(),
        'kinetic_energy': kinetic_energy.tolist(),
        'grad_dvdy': grad_dvdy.tolist(),
        'grad_dudx': grad_dudx.tolist(),
        'uay': uavalueyy.tolist(),
        'ke_dimless': ke_dimless.tolist(),
        **derived
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
    #sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
    #sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
    #sol="/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Coarse_paticle37/case370428_1"
    #sol = "/home/amber/OpenFOAM/amber-v2306/Marino/Run8/case0702_5"
    sol = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_6"

    X, Y, Z = fluidfoam.readmesh(sol)
    times = np.arange(1, 40, 1)  # 对应原来的1-79,步长2
    results = []
    BASE_PATH = '/home/amber/postpro/selecting_variant/'
#FILE_PREFIX = 'case230427_4'  # 修改这里即可自动更新文件名
    #FILE_PREFIX = 'case090429_1'  # 修改这里即可自动更新文件名
    FILE_PREFIX = 'case0704_6'  # 修改这里即可自动更新文件名

    for time_v in times:
        result = process_time_step(sol, time_v, X, Y)
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
