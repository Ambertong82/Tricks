import numpy as np
import pandas as pd
import os
import fluidfoam
from fractions import Fraction
### this code is used to extract the data at specific x location (head-0.3) at every time step #####

# 常量定义
A = 1/4  # 修改这里即可自动更新文件名（支持分数/浮点数）
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
        'uadimless': f'x{a_id}xuadimless.csv',
        'ubdimless': f'x{a_id}xubdimless.csv',
        'shearstress': f'x{a_id}xShearStress.csv',
        'kinetic_energy': f'x{a_id}xKineticEnergy.csv',
        'production': f'x{a_id}xProduction.csv',
        'production_xy': f'x{a_id}xProduction_xy.csv',
        'production_dudx': f'x{a_id}xProduction_dudx.csv',
        'production_dvdy': f'x{a_id}xProduction_dvdy.csv',
        # 'grad_dudx': f'x{a_id}xGrad_dudx.csv',
        # 'grad_dvdy': f'x{a_id}xGrad_dvdy.csv',
        # 'dragpart': f'x{a_id}xDragPart.csv',
        'buoyancy': f'x{a_id}xBuoyancy.csv',
        'dissipation': f'x{a_id}xDissipation.csv',
        'uay': f'x{a_id}xuay.csv',
        'uby': f'x{a_id}xuby.csv',
        'ke_dimless': f'x{a_id}xke_dimless.csv',
        'H': f'x{a_id}xH.csv',
        'U': f'x{a_id}xU.csv',
        'ALPHA': f'x{a_id}xALPHA.csv',
        'H_depth': f'x{a_id}xH_depth.csv',
        'H_alpha': f'x{a_id}xH_alpha.csv',
        'U_alpha': f'x{a_id}xU_alpha.csv',
        'ALPHA_alpha': f'x{a_id}xALPHA_alpha.csv',
        'H_depth_alpha': f'x{a_id}xH_depth_alpha.csv',
        'advection': f'x{a_id}xAdvection.csv',
        'vorticity': f'x{a_id}xVorticity.csv',
        'ycrossing': f'x{a_id}xYcrossing.csv',
        'uhat': f'x{a_id}xu_hat.csv',
        'dragpart1': f'x{a_id}xDragPart1.csv',
        'dragpart2': f'x{a_id}xDragPart2.csv',
        'drag1': f'x{a_id}xDrag1.csv',
        'transport': f'x{a_id}xTransport.csv',
        'FX': f'x{a_id}xFX.csv',    
        'FY': f'x{a_id}xFY.csv',
        'k.b': f'x{a_id}xk_b.csv',
    }


def calculate_derived_values(xvalue,yvalue, alpha, kinetic_energy, gamma, grad_dudx, grad_dvdx, 
                           grad_dudy, grad_dvdy, nutb, uavalue, ubvalue, uavalueyy, 
                           ubvalueyy, grad_alpha1, grad_alpha2, grad_beta1, grad_beta2, omega, grad_vortx, grad_vorty,
                           uavalueorigin,vorticity_z,grad_k1dx,grad_k2dy,dx,dy
                           ):
    """计算衍生量"""
    # rho_mix = alpha * 2217 + 1000
    # drhody = np.gradient(rho_mix, yvalue)
    

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     Rig = -9.81 * drhody / (1000 * grad_dudy**2)
    #     Rigg = -9.81*2217*grad_alpha2 / (1000 * grad_dudy**2)
    #     Rig = np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0)
    #     Rigg = np.nan_to_num(Rigg, nan=0.0, posinf=0.0, neginf=0.0)

    

    
    reynolds12 = nutb * (grad_dudy + grad_dvdx)*1000
    reynolds11 = nutb * (grad_dudx + grad_dvdx-2/3*kinetic_energy)*1000
    reynolds22 = nutb * (grad_dvdy + grad_dvdy-2/3*kinetic_energy)*1000
    center = (yvalue[1:] - yvalue[:-1]) / 2 + yvalue[:-1]
    yplus = np.sqrt(1e-6 * grad_dudy[0]) * center / 1e-6
    shearstress = grad_dudy+grad_dvdx
    production_xy = (1-alpha)*1000*nutb * (grad_dudy**2+grad_dvdx**2+2*grad_dudy*grad_dvdx) 
    production_dudx = (1-alpha)*1000*nutb * (2*grad_dudx**2)- (1-alpha)*2/3*1000*kinetic_energy *grad_dudx
    production_dvdy = (1-alpha)*1000*nutb * (2*grad_dvdy**2)- (1-alpha)*2/3*1000*kinetic_energy *grad_dvdy
    production = production_xy + production_dudx + production_dvdy
    # dragpart = gamma * ((ubvalue-uavalue)*grad_alpha1 + (ubvalueyy-uavalueyy)*grad_alpha2)* nutb /(1-alpha)

    tauxx = 1e-3*grad_dudx*2
    tauxy = 1e-3*(grad_dudy+grad_dvdx)
    tauyy = 1e-3*grad_dvdy*2
    seoxx = grad_beta1**2/(1-alpha)
    seoxy = (grad_beta1*grad_beta2)/(1-alpha)
    seoyy = grad_beta2**2/(1-alpha)
    buoyancy = nutb*(tauxx*seoxx + 2*tauxy*seoxy + tauyy*seoyy)
     
    dissipation = -(1-alpha)*1000*0.09*kinetic_energy*omega
    drag1 = gamma*nutb*1/(1-alpha)*(grad_alpha1*(ubvalue-uavalue)+grad_alpha2*(ubvalueyy - uavalueyy))
    Fx =  (1-alpha)*1000*(1e-6+nutb*0.6)*grad_k1dx
    Fy =  (1-alpha)*1000*(1e-6+nutb*0.6)*grad_k2dy
    dFx_dx = np.gradient(Fx) / dx  # ∂Fx/∂x
    dFy_dy = np.gradient(Fy) / dy # ∂Fy/∂y
    transport = dFx_dx + dFy_dy
################################################################################

    
    ### calculating the vorticity related quantities ####

    advection = (1-alpha) * 1000 *(ubvalue * grad_vortx + ubvalueyy * grad_vorty)
    drag_part1 = -gamma*grad_alpha2*(ubvalue-uavalue)+gamma*grad_alpha1*(ubvalueyy - uavalueyy)
    drag_part2 = -alpha*gamma*vorticity_z

        # Find the y-coordinate where alpha crosses below 1e-5
        # 首先筛选出 y > 0.005 的点
    y_threshold = 0.005
    valid_mask = yvalue > y_threshold

    # if np.any(valid_mask):
    #         # 在有效范围内寻找 alpha < 1e-5 的点
    #         alpha_threshold = 1e-5
    #         below_threshold = (alpha[valid_mask] < alpha_threshold)
            
    #         if np.any(below_threshold):
    #             # 找到第一个满足条件的点的相对索引
    #             first_below_rel_index = np.argmax(below_threshold)
    #             # 转换为原始数组中的绝对索引
    #             valid_indices = np.where(valid_mask)[0]
    #             max_ya_crossing_index_alpha = valid_indices[first_below_rel_index]
    #             y_crossing_alpha = yvalue[max_ya_crossing_index_alpha]  # 修正变量名
    #             u_crossing_alpha = uavalue[max_ya_crossing_index_alpha]
    #             #print(f"Found y_crossing at {y_crossing_alpha} for xx={xx}")
    #         else:
    #             # 如果没有找到，使用有效范围内的最大y值
    #             max_ya_crossing_index_alpha = np.where(valid_mask)[0][-1]
    #             y_crossing_alpha = yvalue[max_ya_crossing_index_alpha]  # 修正变量名
    #             u_crossing_alpha = uavalue[max_ya_crossing_index_alpha]
    #             #print(f"No alpha < {alpha_threshold} found above y={y_threshold} for xx={xx}, using max y={y_crossing_alpha}")
    # else:
    #         # 如果没有y>0.005的点，使用最后一个点
    #         max_ya_crossing_index_alpha = len(yvalue) - 1
    #         y_crossing_alpha = yvalue[max_ya_crossing_index_alpha]  # 修正变量名
    #         u_crossing_alpha = uavalue[max_ya_crossing_index_alpha]  # 无法定义

    #     # Vectorized integration
    # ua_alpha_alpha = uavalue * alpha
    # integral_alpha = np.trapz(ua_alpha_alpha[:max_ya_crossing_index_alpha], yvalue[:max_ya_crossing_index_alpha])
    # integralU_alpha = np.trapz(uavalue[:max_ya_crossing_index_alpha], yvalue[:max_ya_crossing_index_alpha])
    # integralU2_alpha = np.trapz(uavalue[:max_ya_crossing_index_alpha]**2, yvalue[:max_ya_crossing_index_alpha])
    # integral2_alpha = np.trapz((uavalue[:max_ya_crossing_index_alpha] * alpha[:max_ya_crossing_index_alpha])**2, yvalue[:max_ya_crossing_index_alpha])
        
    #     # Calculate derived quantities
    # U_alpha = integralU2_alpha / integralU_alpha if integralU_alpha != 0 else 0
    # H_alpha = integral_alpha**2 / integral2_alpha if integral2_alpha != 0 else 0
    # ALPHA_alpha = integral_alpha / integralU_alpha if integralU_alpha != 0 else 0
    # H_depth_alpha = integralU_alpha**2 / integralU2_alpha if integralU2_alpha != 0 else 0
                # 寻找速度正负交界点
    # mask = (X == closest_x) & (Y >= 0) & (alpha > alpha_threshold)
    mask2 = alpha > alpha_threshold
    uavalue = uavalue[mask2]  
    yvalue = yvalue[mask2]     
    alpha = alpha[mask2]  
    sign_changes = np.where(np.diff(np.sign(uavalue)))[0]
        # 找出第一个（最低处）满足 "负→正" 的变化点
    for idx in sign_changes:
        if yvalue[idx] > 0.001 and uavalue[idx] > 0 and uavalue[idx + 1] < 0: #and alpha[idx] > 1e-5:
            max_ya_crossing_index = idx + 1  # （可选 +1，取决于是否需要变化后的位置）
            break
    else:
            max_ya_crossing_index = len(yvalue) - 1  # 没找到则默认取最高处
    y_crossing = yvalue[max_ya_crossing_index]
    u_crossing = uavalue[max_ya_crossing_index]
    print(f"Found y_crossing at {y_crossing} for xx={xvalue[0]}at max_ya_crossing_index={max_ya_crossing_index}")
        
    # Vectorized integration
    ua_alpha = uavalue * alpha
    Ucih = np.trapz(ua_alpha[:max_ya_crossing_index], yvalue[:max_ya_crossing_index])
    Uh = np.trapz(uavalue[:max_ya_crossing_index], yvalue[:max_ya_crossing_index])
    U2h = np.trapz(uavalue[:max_ya_crossing_index]**2, yvalue[:max_ya_crossing_index])
    Uci2h = np.trapz((uavalue[:max_ya_crossing_index] * alpha[:max_ya_crossing_index])**2, yvalue[:max_ya_crossing_index])
        
    # Calculate derived quantities
    U = U2h / Uh if Uh != 0 else 0
    # H = Ucih**2 / Uci2h if Uci2h != 0 else 0
    ALPHA = Ucih / Uh if Uh != 0 else 0
    H_depth = Uh**2 / U2h if U2h != 0 else 0

    u_hat = uavalueorigin - U 
    kinetic_energy = 1e4*kinetic_energy


    return {
        # 'Rig': Rig.tolist(),
        # 'Rigg': Rigg.tolist(),
        # 'omegaz': omegaz.tolist(),
        'reynolds12': reynolds12.tolist(),
        'reynolds11': reynolds11.tolist(),
        'reynolds22': reynolds22.tolist(),
        'yplus': yplus.tolist(),
        'grad_dudx': grad_dudx.tolist(),
        'grad_dvdy': grad_dvdy.tolist(),
        'alpha': alpha.tolist(),
        'shearstress': shearstress.tolist(),
        'production': production.tolist(),
        'production_xy': production_xy.tolist(),
        'production_dudx': production_dudx.tolist(),
        'production_dvdy': production_dvdy.tolist(),
        # 'dragpart': dragpart.tolist(),
        'buoyancy': buoyancy.tolist(),
        'dissipation': dissipation.tolist(),

        # 'H': [H],
        'U': [U],
        'ALPHA': [ALPHA],
        'H_depth': [H_depth],
        # 'H_alpha': [H_alpha],
        # 'U_alpha': [U_alpha],
        # 'ALPHA_alpha': [ALPHA_alpha],
        # 'H_depth_alpha': [H_depth_alpha],
        'advection': advection.tolist(),
        'ycrossing': [y_crossing],
        'uhat': u_hat.tolist(),
        'dragpart1': drag_part1.tolist(),
        'dragpart2': drag_part2.tolist(),
        'drag1': drag1.tolist(),
        'transport': transport.tolist(),
        'FX': Fx.tolist(),
        'FY': Fy.tolist(),

    
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
    grad_Ub = fluidfoam.readtensor(sol, str(time_v), "grad(U.b)")
    grad_alpha = fluidfoam.readvector(sol, str(time_v), "grad(alpha.a)")
    grad_beta = fluidfoam.readvector(sol, str(time_v), "grad(alpha.b)")
    gamma = fluidfoam.readscalar(sol, str(time_v), "K")
    vorticity_grad = fluidfoam.readtensor(sol, str(time_v), "grad(vorticity)")
    vorticity = fluidfoam.readvector(sol, str(time_v), "vorticity")
    grad_k = fluidfoam.readvector(sol, str(time_v), "grad(k.b)")

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
    grad_dudx = grad_Ub[0][mask]
    grad_dudy = grad_Ub[3][mask]
    grad_dvdx = grad_Ub[1][mask]
    grad_dvdy = grad_Ub[4][mask]
    grad_alpha1 = grad_alpha[0][mask]
    grad_alpha2 = grad_alpha[1][mask]
    grad_beta1 = grad_beta[0][mask]
    grad_beta2 = grad_beta[1][mask]
    grad_vortx = vorticity_grad[2][mask]
    grad_vorty = vorticity_grad[5][mask]
    vorticity_z = vorticity[2][mask]
    grad_k1dx = grad_k[0][mask]
    grad_k2dy = grad_k[1][mask]
    dx = dx[mask]
    dy = dy[mask]

    # 计算衍生量
    derived = calculate_derived_values(
        xvalue,yvalue, alpha_value, kinetic_energy, gamma[mask], grad_dudx, grad_dvdx,
        grad_dudy, grad_dvdy, nutb_value, uavalue, ubvalue, uavalueyy, ubvalueyy,
        grad_alpha1, grad_alpha2, grad_beta1, grad_beta2, omega_value,grad_vortx, grad_vorty,
        uavalueorigin, vorticity_z,grad_k1dx,grad_k2dy,dx,dy
    )

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
        'grad_dvdy': grad_dvdy.tolist(),
        'grad_dudx': grad_dudx.tolist(),
        'uay': uavalueyy.tolist(),
        'uby': ubvalueyy.tolist(),
        'ke_dimless': ke_dimless.tolist(),
        'k.b': kb.tolist(),
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
    #sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4coarse"
    #sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
    # sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090912_1"
    #sol="/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Coarse_paticle37/case370428_1"
    #sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Large_particle53/case530826_4"
    sol = "/media/amber/53EA-E81F/PhD/case231020_6"

    X, Y, Z = fluidfoam.readmesh(sol)
    dx = np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=0)
    times = np.arange(4, 15, 1)  # 对应原来的1-79,步长2
    results = []
    BASE_PATH = '/home/amber/postpro/selecting_variant/'
    # FILE_PREFIX = 'case230427_4midd'  # 修改这里即可自动更新文件名
    # FILE_PREFIX = 'case090912_1'  # 修改这里即可自动更新文件名
    #FILE_PREFIX = 'case530628_1'  # 修改这里即可自动更新文件名
    FILE_PREFIX = 'case231020_6'  # 修改这里即可自动更新文件名

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
