import numpy as np
import pandas as pd
import os
import fluidfoam
from fractions import Fraction
### this code is used to extract the data at specific x location (head-0.3) at every time step #####

# 常量定义

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
        'U': f'x{a_id}xU.csv',
        'H_depth': f'x{a_id}xH_depth.csv',
        'ycrossing': f'x{a_id}xycrossing.csv',
        'Udir': f'x{a_id}xUdir.csv',
        'u_max_velocity': f'x{a_id}xu_max_velocity.csv',
        'y_max_velocity': f'x{a_id}xy_max_velocity.csv',
        # 新增拟合结果文件
        'av_fitted': f'x{a_id}xav_fitted.csv',
        'beta_fitted': f'x{a_id}xbeta_fitted.csv',
        'gamma_fitted': f'x{a_id}xgamma_fitted.csv',
        'r_squared_below': f'x{a_id}xr_squared_below.csv',
        'r_squared_above': f'x{a_id}xr_squared_above.csv',
    }

def calculate_derived_values( 
        xvalue,yvalue, alpha, uavalue, 
          ):
    """计算衍生量"""



    mask2 = alpha > alpha_threshold
    uavalue = uavalue[mask2]  
    yvalue = yvalue[mask2]     
    alpha = alpha[mask2]  

    # 找到速度最大值的y值
    max_u_index = np.argmax(uavalue)  # 速度最大值的索引
    y_max_velocity = yvalue[max_u_index]
    u_max_velocity = uavalue[max_u_index]
    # 新增：对umax对应的ymax以下位置进行拟合
    # 选择ymax以下的数据点
    # ==================== 拟合1: ymax以下位置 ====================
    # 公式: (z/ymax)^(1/av) = u/umax
    mask_below_max = yvalue <= y_max_velocity
    y_below = yvalue[mask_below_max]
    u_below = uavalue[mask_below_max]
    

    
    av_fitted = 0.0
    r_squared_below = 0.0
    
    if len(y_below) > 2:  # 需要有足够的数据点进行拟合
        y_normalized = y_below / y_max_velocity
        u_normalized = u_below / u_max_velocity
        
        
        if len(y_normalized) > 2:
            # 对公式取对数进行线性拟合
            # (1/av) * ln(y/ymax) = ln(u/umax)
            X_fit = np.log(y_normalized)
            Y_fit = np.log(u_normalized)
            
            # 线性回归拟合
            slope, intercept = np.polyfit(X_fit, Y_fit, 1)
            
            # 计算 R²
            y_pred = slope * X_fit + intercept
            ss_res = np.sum((Y_fit - y_pred) ** 2)
            ss_tot = np.sum((Y_fit - np.mean(Y_fit)) ** 2)
            r_squared_below = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # av = 1/slope
            if abs(slope) > 1e-6:
                av_fitted = 1.0 / slope
            else:
                av_fitted = 0.0
            
            print(f"ymax以下拟合结果: av = {av_fitted:.4f}, R² = {r_squared_below:.4f}")

    # ==================== 拟合2: ymax以上位置 ====================
    # 公式: u/umax = exp[-beta(z-Hm)/(H-Hm)^gamma]
    mask_above_max = (yvalue > y_max_velocity) & (uavalue > 0)
    y_above = yvalue[mask_above_max]
    u_above = uavalue[mask_above_max]
    
    
    
    beta_fitted = 0.0
    gamma_fitted = 1.0
    r_squared_above = 0.0
    
    if len(y_above) > 2:
        # 这里Hm是最大速度位置y_max_velocity，H是流动高度（需要定义）
        # 假设H是y的最大值（或者使用您之前定义的y_crossing）
        H = np.max(yvalue) # 或者使用 y_crossing
        
        
        # 避免分母为0
        if H > y_max_velocity:
            u_normalized_above = u_above / u_max_velocity
            z_minus_Hm = y_above - y_max_velocity
            H_minus_Hm = H - y_max_velocity
            
            # 对公式取对数进行拟合
            # ln(u/umax) = -beta * ((z-Hm) / (H-Hm))^gamma
            # 这需要非线性拟合，我们使用scipy的curve_fit
            
            try:
                from scipy.optimize import curve_fit
                
                def above_fit_func(z, beta, gamma):
                    return -beta * ((z - y_max_velocity) / (H_minus_Hm) )** gamma
                
                # 初始猜测值
                p0 = [1.0, 1.0]
                
                # 非线性最小二乘拟合
                Y_fit_above = np.log(u_normalized_above)
                popt, pcov = curve_fit(above_fit_func, y_above, Y_fit_above, p0=p0, maxfev=5000)
                
                beta_fitted, gamma_fitted = popt
                
                # 计算R²
                y_pred_above = above_fit_func(y_above, *popt)
                ss_res_above = np.sum((Y_fit_above - y_pred_above) ** 2)
                ss_tot_above = np.sum((Y_fit_above - np.mean(Y_fit_above)) ** 2)
                r_squared_above = 1 - (ss_res_above / ss_tot_above) if ss_tot_above != 0 else 0
                
                print(f"ymax以上拟合结果: beta = {beta_fitted:.4f}, gamma = {gamma_fitted:.4f}, R² = {r_squared_above:.4f}")
                
            except ImportError:
                print("警告: 需要安装scipy进行非线性拟合: pip install scipy")
            except Exception as e:
                print(f"ymax以上拟合失败: {e}")

    # fit above umax_velocity
    

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

    Udir = np.trapz(uavalue[:max_ya_crossing_index], yvalue[:max_ya_crossing_index])
    Uh = np.trapz(uavalue[:max_ya_crossing_index], yvalue[:max_ya_crossing_index])
    U2h = np.trapz(uavalue[:max_ya_crossing_index]**2, yvalue[:max_ya_crossing_index])

    # Calculate derived quantities
    U = U2h / Uh if Uh != 0 else 0
    Udir = Udir/yvalue[max_ya_crossing_index] if yvalue[max_ya_crossing_index] !=0 else 0 

    
    H_depth = Uh**2 / U2h if U2h != 0 else 0


    return {
        'alpha': alpha.tolist(),
        'U': [U],
        'H_depth': [H_depth],
        'Udir': [Udir],
        'u_max_velocity': [u_max_velocity],
        'y_max_velocity': [y_max_velocity],
        'ycrossing': [y_crossing],
        # 新增拟合结果
        'av_fitted': [av_fitted],
        'beta_fitted': [beta_fitted],
        'gamma_fitted': [gamma_fitted],
        'r_squared_below': [r_squared_below],
        'r_squared_above': [r_squared_above],
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
    nuFra = fluidfoam.readscalar(sol, str(time_v), "nuFra")


    

    # z_mask = np.isclose(Z, 0.255)  # 或用 Z == 0（如果数据是精确的）
    z_mask = np.isclose(Z,z0)
    Ua = Ua[:, z_mask]
    Ub = Ub[:, z_mask]
    alpha = alpha[z_mask]
    nutb = nutb[z_mask]
    omegab = omegab[z_mask]
    vorticity = vorticity[:,z_mask]
    gradUa = gradUa[:,z_mask]
 


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

    vorticity = vorticity[2][mask]  # 提取z方向的涡量分量


    dx = dx[mask]
    dy = dy[mask]



    # 计算衍生量
    derived = calculate_derived_values(
        xvalue,yvalue, alpha_value, uavalue, 
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
        'uay': uavalueyy.tolist(),
        'uby': ubvalueyy.tolist(),

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
    ############### 主程序 ##################
    # sol = "/media/amber/53EA-E81F/PhD/case231020_5"
    sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case091020_5"
  

    X, Y, Z = fluidfoam.readmesh(sol)
    z0 = 0.135
    # z0 = 0.1
    # 2. 提取 Z=0 的平面（浮点数比较用 np.isclose）
    z_mask = np.isclose(Z, z0)  # 或用 Z == 0（如果数据是精确的）
    X = X[z_mask]
    Y = Y[z_mask]
    dx = np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=0)
    times = np.arange(4, 15, 1)  # 对应原来的1-79,步长2
    results = []
    BASE_PATH = '/home/amber/postpro/selecting_variant/'
    # FILE_PREFIX = 'case231020_5middle'  # 修改这里即可自动更新文件名
    FILE_PREFIX = 'case091020_5middle'  # 修改这里即可自动更新文件名

    
    # === 定义A值的列表 ===
    a_values = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 1)]  # 1/2, 1/3, 1/4, 1
    
    # === 对每个A值进行循环 ===
    for a_val in a_values:
        global A
        A = float(a_val)  # 转换为浮点数用于计算
        
        print(f"\n=== 处理 A = {a_val} ===")
        
        # 为当前A值创建新的结果列表
        results = []
        
        for time_v in times:
            result = process_time_step(sol, time_v, X, Y, Z, dx, dy, z0)
            if result:
                results.append(result)
        
        if results:
            save_data(results, BASE_PATH, FILE_PREFIX)
            print(f"A = {a_val} 数据处理完成，结果已保存到以下文件：")
            for name in get_output_files().values():
                print(f"  - {FILE_PREFIX}_{name}")
        else:
            print(f"A = {a_val} 未找到有效数据")

if __name__ == '__main__':
    main()
