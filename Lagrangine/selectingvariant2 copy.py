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
        'G': f'x{a_id}xG.csv',
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
        'vorticity': f'x{a_id}xvorticity.csv',
    }


def calculate_derived_values( 
        xvalue,yvalue, alpha, kinetic_energy,
       nutb, uavalue, ubvalue, uavalueyy, ubvalueyy,
        omega_value,
        uavalueorigin,dx,dy,grad_dudx,grad_dvdx,grad_dudy,grad_dvdy,grad_dudz,grad_dvdz
                           ):
    """计算衍生量"""
    # ############################################################### #
    # ##                   ↓↓↓ 添加调试代码 ↓↓↓                    ## #
    # print("--- Debugging Shapes in calculate_derived_values ---")
    # print(f"alpha.shape: {alpha.shape}")
    # print(f"nutb.shape: {nutb.shape}")
    # print(f"kinetic_energy.shape: {kinetic_energy.shape}")
    # print(f"grad_dudy.shape: {grad_dudy.shape}")
    # print(f"grad_dvdx.shape: {grad_dvdx.shape}")
    # print(f"grad_dudx.shape: {grad_dudx.shape}")
    # print(f"grad_dvdy.shape: {grad_dvdy.shape}")
    # print("----------------------------------------------------")
    # ############################################################### #
    # rho_mix = alpha * 2217 + 1000
    # drhody = np.gradient(rho_mix, yvalue)
    

    kinetic_energy = kinetic_energy
    production_xy = (1-alpha)*1000*nutb * (grad_dudy+grad_dvdx)**2+grad_dudz**2+grad_dvdz**2
    production_dudx = (1-alpha)*1000*nutb * (2*grad_dudx**2)-2/3*(1-alpha)*1000*nutb*(grad_dudx+grad_dvdy)**2
    production_dvdy = (1-alpha)*1000*nutb * (2*grad_dvdy**2)-2/3*(1-alpha)*1000*nutb*(grad_dudx+grad_dvdy)**2
    production = production_xy + production_dudx + production_dvdy
    


    # dragpart = gamma * ((ubvalue-uavalue)*grad_alpha1 + (ubvalueyy-uavalueyy)*grad_alpha2)* nutb /(1-alpha)

################################################################################

    
    ### calculating the vorticity related quantities ####


        # Find the y-coordinate where alpha crosses below 1e-5
        # 首先筛选出 y > 0.005 的点


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



    return {
        # 'Rig': Rig.tolist(),
        # 'Rigg': Rigg.tolist(),
        # 'omegaz': omegaz.tolist(),
 
        'alpha': alpha.tolist(),
        'production': production.tolist(),
 

        # 'H': [H],
        'U': [U],
        'ALPHA': [ALPHA],
        'H_depth': [H_depth],
        # 'H_alpha': [H_alpha],
        # 'U_alpha': [U_alpha],
        # 'ALPHA_alpha': [ALPHA_alpha],
        # 'H_depth_alpha': [H_depth_alpha],
   
        'ycrossing': [y_crossing],
        'uhat': u_hat.tolist(),


    
    }

def process_time_step(sol, time_v, X, Y,Z,dx,dy,z0):
    """处理单个时间步"""
    # 读取场数据
    
    Ua = fluidfoam.readvector(sol, str(time_v), "U.a")
    Ub = fluidfoam.readvector(sol, str(time_v), "U.b")
    alpha = fluidfoam.readscalar(sol, str(time_v), "alpha.a")
    nutb = fluidfoam.readscalar(sol, str(time_v), "nut.b")
    kb = fluidfoam.readscalar(sol, str(time_v), "k.b")
    omegab = fluidfoam.readscalar(sol, str(time_v), "omega.b")
    vorticity = fluidfoam.readvector(sol, str(time_v), "vorticity")
    gradUa = fluidfoam.readtensor(sol, str(time_v), "grad(U.a)")

    G = 2*(gradUa[0]**2 + gradUa[4]**2 + gradUa[8]**2) + \
        (gradUa[1]+gradUa[3])**2 + (gradUa[2]+gradUa[6])**2 + (gradUa[5]+gradUa[7])**2 -\
        -2/3*(gradUa[0]+gradUa[4]+gradUa[8])**2
    

    # z_mask = np.isclose(Z, 0.255)  # 或用 Z == 0（如果数据是精确的）
    z_mask = np.isclose(Z,z0)
    Ua = Ua[:, z_mask]
    Ub = Ub[:, z_mask]
    alpha = alpha[z_mask]
    nutb = nutb[z_mask]
    kb = kb[z_mask]
    omegab = omegab[z_mask]
    vorticity = vorticity[:,z_mask]
    gradUa = gradUa[:,z_mask]
    G = G[z_mask]


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
    vorticity = vorticity[2][mask]  # 提取z方向的涡量分量
    grad_dudx = gradUa[0][mask]
    grad_dvdx = gradUa[1][mask]
    grad_dudy = gradUa[3][mask]
    grad_dvdy = gradUa[4][mask]
    grad_dudz = gradUa[6][mask]
    grad_dvdz = gradUa[7][mask]
 


    dx = dx[mask]
    dy = dy[mask]



    # 计算衍生量
    derived = calculate_derived_values(
        xvalue,yvalue, alpha_value, kinetic_energy,
       nutb_value, uavalue, ubvalue, uavalueyy, ubvalueyy,
        omega_value,
        uavalueorigin,dx,dy,grad_dudx,grad_dvdx,grad_dudy,grad_dvdy,grad_dudz,grad_dvdz
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
        'vorticity': vorticity.tolist(),
        'uay': uavalueyy.tolist(),
        'uby': ubvalueyy.tolist(),
        'ke_dimless': ke_dimless.tolist(),
        'k.b': kb.tolist(),
        'G': G.tolist(),
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
    sol = "/media/amber/53EA-E81F/PhD/case231020_5"
    # sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/3D/case231020_5"

    X, Y, Z = fluidfoam.readmesh(sol)
    z0 = 0.065
    # 2. 提取 Z=0 的平面（浮点数比较用 np.isclose）
    z_mask = np.isclose(Z, z0)  # 或用 Z == 0（如果数据是精确的）
    X = X[z_mask]
    Y = Y[z_mask]
    dx = np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=0)
    times = np.arange(4, 12, 1)  # 对应原来的1-79,步长2
    results = []
    BASE_PATH = '/home/amber/postpro/selecting_variant/'
    # FILE_PREFIX = 'case230427_4midd'  # 修改这里即可自动更新文件名
    # FILE_PREFIX = 'case090912_1'  # 修改这里即可自动更新文件名
    #FILE_PREFIX = 'case530628_1'  # 修改这里即可自动更新文件名
    FILE_PREFIX = 'case231020_5side14'  # 修改这里即可自动更新文件名

    for time_v in times:
        result = process_time_step(sol, time_v, X, Y, Z, dx,dy,z0)
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
