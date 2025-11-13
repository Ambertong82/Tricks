import numpy as np
import pandas as pd
import os
import fluidfoam
from fractions import Fraction
import matplotlib.pyplot as plt
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

def fit_velocity_profile(yvalue, uavalue, A, save_plot, plot_dir, time_v):

    mask = uavalue >= 0
    yvalue = yvalue[mask]
    uavalue = uavalue[mask]

    # 找到速度最大值的y值
    max_u_index = np.argmax(uavalue)  # 速度最大值的索引
    y_max_velocity = yvalue[max_u_index]
    u_max_velocity = uavalue[max_u_index]
    # 新增：对umax对应的ymax以下位置进行拟合

    
   # ==================== 拟合1: ymax以下位置 ====================
    # 初始化变量
    fitted_below_data = None
    fitted_above_data = None

    mask_below_max = yvalue <= y_max_velocity
    y_below = yvalue[mask_below_max]
    u_below = uavalue[mask_below_max]

    av_fitted = 0.0
    r_squared_below = 0.0

    # 添加基础验证
    if len(y_below) > 2 and y_max_velocity > 0 and u_max_velocity > 0:
        y_normalized = y_below / y_max_velocity
        u_normalized = u_below / u_max_velocity
        
        # 确保所有值都大于0且合理
        valid_mask = (y_normalized > 1e-6) & (u_normalized > 1e-6) & (y_normalized <= 1.0)
        y_normalized = y_normalized[valid_mask]
        u_normalized = u_normalized[valid_mask]
        
        if len(y_normalized) > 2:
            try:
                X_fit = np.log(y_normalized)
                Y_fit = np.log(u_normalized)
                
                # 方法1: 强制过原点拟合（更符合物理意义）
                # Y = m*X + 0, 其中 m = 1/av
                slope = np.sum(X_fit * Y_fit) / np.sum(X_fit**2)
                intercept = 0.0  # 强制截距为0
                
                # 方法2: 如果你想用polyfit但检查截距（备选方案）
                # slope_poly, intercept_poly = np.polyfit(X_fit, Y_fit, 1)
                # 如果截距很小，可以使用polyfit结果
                # if abs(intercept_poly) < 0.1:  # 阈值可根据实际情况调整
                #     slope = slope_poly
                #     intercept = intercept_poly
                
                # 计算 R²（使用过原点模型的R²）
                y_pred = slope * X_fit  # 注意：这里用 slope*X_fit + 0
                ss_res = np.sum((Y_fit - y_pred) ** 2)
                ss_tot = np.sum((Y_fit - np.mean(Y_fit)) ** 2)
                r_squared_below = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # av = 1/slope
                if abs(slope) > 1e-6:
                    av_fitted = 1.0 / slope
                else:
                    av_fitted = 0.0
                
                print(f"ymax以下拟合结果: av = {av_fitted:.4f}, R² = {r_squared_below:.4f}")
                print(f"斜率: {slope:.4f}, 截距: {intercept:.4f}")
                
                # 检查边界条件：当y=Hm时，u是否接近umax
                boundary_check = f"边界检查: y/Hm=1时，u/umax应接近1，实际为{np.exp(intercept):.4f}"
                print(boundary_check)
                
                # 保存拟合数据用于绘图
                fitted_below_data = {
                    'y_original': y_below,
                    'u_original': u_below,
                    'y_normalized': y_normalized,
                    'u_normalized': u_normalized,
                    'slope': slope,
                    'intercept': intercept,
                    'av': av_fitted,
                    'r_squared': r_squared_below
                }
                
            except Exception as e:
                print(f"ymax以下拟合失败: {e}")
                fitted_below_data = None
    else:
        print("ymax以下数据不足或参数无效")

 
 #==================== 拟合2: ymax以上位置 ====================
    mask_above_max = (yvalue > y_max_velocity) 
    y_above = yvalue[mask_above_max]
    u_above = uavalue[mask_above_max]

    beta_fitted = 0.0
    gamma_fitted = 1.0
    r_squared_above = 0.0
    fitted_above_data = None

    # 存储两种方法的结果
    fit_results = {'original': None, 'log': None}

    if len(y_above) > 2:
        H = np.max(yvalue)
        
        if H > y_max_velocity:
            u_normalized_above = u_above / u_max_velocity
            H_minus_Hm = H - y_max_velocity
            normalized_height = (y_above - y_max_velocity) / H_minus_Hm
            
            # 确保数据有效性
            valid_mask = (u_normalized_above >= 0) & (u_normalized_above <= 1.0) & (normalized_height > 0) & (normalized_height <= 1.0)
            X_fit_above = normalized_height[valid_mask]  # 归一化高度作为X
            Y_fit_above = u_normalized_above[valid_mask]  # 归一化速度作为Y
            
            if len(X_fit_above) > 2:
                try:
                    from scipy.optimize import curve_fit
                    
                    # 定义两种拟合函数
                    def above_fit_func_original(norm_h, beta, gamma):
                        """原始空间拟合函数"""
                        norm_h = np.clip(norm_h, 1e-6, 1.0)
                        return np.exp(-beta * (norm_h ** gamma))
                    
                    def above_fit_func_log(norm_h, beta, gamma):
                        """对数空间拟合函数：返回 ln(u/umax)"""
                        norm_h = np.clip(norm_h, 1e-6, 1.0)
                        return -beta * (norm_h ** gamma)
                    
                    p0 = [2.0, 2.0]
                    bounds = ([0.1, 0.1], [20.0, 10.0])
                    
                    # ========== 方法1: 原始空间拟合 ==========
                    try:
                        popt_original, pcov_original = curve_fit(
                            above_fit_func_original, X_fit_above, Y_fit_above, 
                            p0=p0, maxfev=5000, bounds=bounds
                        )
                        beta_original, gamma_original = popt_original
                        
                        # 计算R²（原始空间）
                        y_pred_original = above_fit_func_original(X_fit_above, *popt_original)
                        ss_res_original = np.sum((Y_fit_above - y_pred_original) ** 2)
                        ss_tot_original = np.sum((Y_fit_above - np.mean(Y_fit_above)) ** 2)
                        r2_original = 1 - (ss_res_original / ss_tot_original) if ss_tot_original != 0 else 0
                        
                        fit_results['original'] = {
                            'beta': beta_original,
                            'gamma': gamma_original,
                            'r_squared': r2_original,
                            'method': 'original_space'
                        }
                        print(f"原始空间拟合: beta = {beta_original:.4f}, gamma = {gamma_original:.4f}, R² = {r2_original:.4f}")
                        
                    except Exception as e:
                        print(f"原始空间拟合失败: {e}")
                        fit_results['original'] = None
                    
                    # ========== 方法2: 对数空间拟合 ==========
                    try:
                        Y_fit_log = np.log(Y_fit_above + 1e-6)  # 避免log(0)
                        
                        popt_log, pcov_log = curve_fit(
                            above_fit_func_log, X_fit_above, Y_fit_log, 
                            p0=p0, maxfev=5000, bounds=bounds
                        )
                        beta_log, gamma_log = popt_log
                        
                        # 计算R²（在对数空间计算，但为了比较，也在原始空间计算一次）
                        y_pred_log_space = above_fit_func_log(X_fit_above, *popt_log)
                        y_pred_original_space = np.exp(y_pred_log_space)  # 转换回原始空间
                        
                        # 在对数空间的R²
                        ss_res_log = np.sum((Y_fit_log - y_pred_log_space) ** 2)
                        ss_tot_log = np.sum((Y_fit_log - np.mean(Y_fit_log)) ** 2)
                        r2_log_space = 1 - (ss_res_log / ss_tot_log) if ss_tot_log != 0 else 0
                        
                        # 在原始空间的R²（用于比较）
                        ss_res_original = np.sum((Y_fit_above - y_pred_original_space) ** 2)
                        ss_tot_original = np.sum((Y_fit_above - np.mean(Y_fit_above)) ** 2)
                        r2_original_compare = 1 - (ss_res_original / ss_tot_original) if ss_tot_original != 0 else 0
                        
                        fit_results['log'] = {
                            'beta': beta_log,
                            'gamma': gamma_log,
                            'r_squared_log_space': r2_log_space,
                            'r_squared_original_space': r2_original_compare,
                            'method': 'log_space'
                        }
                        print(f"对数空间拟合: beta = {beta_log:.4f}, gamma = {gamma_log:.4f}")
                        print(f"  - 对数空间R² = {r2_log_space:.4f}, 原始空间R² = {r2_original_compare:.4f}")
                        
                    except Exception as e:
                        print(f"对数空间拟合失败: {e}")
                        fit_results['log'] = None
                    
                    # ========== 选择最佳拟合结果 ==========
                    if fit_results['original'] and fit_results['log']:
                        # 比较两种方法的原始空间R²，选择更好的
                        r2_orig = fit_results['original']['r_squared']
                        r2_log = fit_results['log']['r_squared_original_space']
                        
                        if r2_orig >= r2_log:
                            best_fit = fit_results['original']
                            print("选择原始空间拟合结果（R²更高）")
                        else:
                            best_fit = fit_results['log']
                            print("选择对数空间拟合结果（R²更高）")
                            
                        beta_fitted = best_fit['beta']
                        gamma_fitted = best_fit['gamma']
                        r_squared_above = max(r2_orig, r2_log)
                        
                    elif fit_results['original']:
                        best_fit = fit_results['original']
                        beta_fitted = best_fit['beta']
                        gamma_fitted = best_fit['gamma']
                        r_squared_above = best_fit['r_squared']
                        print("使用原始空间拟合结果（对数空间拟合失败）")
                        
                    elif fit_results['log']:
                        best_fit = fit_results['log']
                        beta_fitted = best_fit['beta']
                        gamma_fitted = best_fit['gamma']
                        r_squared_above = best_fit['r_squared_original_space']
                        print("使用对数空间拟合结果（原始空间拟合失败）")
                    else:
                        print("两种拟合方法都失败")
                        fitted_above_data = None
                        return beta_fitted, gamma_fitted, r_squared_above, fitted_above_data
                    
                    # 最终结果
                    print(f"最终选择: beta = {beta_fitted:.4f}, gamma = {gamma_fitted:.4f}, R² = {r_squared_above:.4f}")
                    
                    # 边界验证
                    u_at_Hmax = np.exp(-beta_fitted * (0.0 ** gamma_fitted))  # 应为1.0
                    u_at_H = np.exp(-beta_fitted * (1.0 ** gamma_fitted))     # 应接近0
                    print(f"边界验证: u(Hmax) = {u_at_Hmax:.4f}, u(H) = {u_at_H:.4f}")
                    
                    # 保存所有拟合数据
                    fitted_above_data = {
                        'normalized_height': X_fit_above,
                        'u_normalized': Y_fit_above,
                        'y_original': y_above[valid_mask],
                        'u_original': u_above[valid_mask],
                        'beta': beta_fitted,
                        'gamma': gamma_fitted,
                        'H_minus_Hm': H_minus_Hm,
                        'y_max_velocity': y_max_velocity,
                        'r_squared': r_squared_above,
                        'fit_results': fit_results  # 保存两种方法的结果
                    }
                    
                except Exception as e:
                    print(f"ymax以上拟合失败: {e}")
                    fitted_above_data = None

        

    # fit above umax_velocity
    # ==================== 绘图 ====================
    if save_plot:
        # 创建绘图目录
        plot_dir = "/home/amber/postpro/fitting_coarse_tc3dmiddle"
        os.makedirs(plot_dir, exist_ok=True)
        a_id = get_a_identifier(A) # 例如A=10/12→"1012"
        
        # # 创建图形
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # # 子图1: ymax以下拟合（对数坐标）
        # ax1.scatter(uavalue / u_max_velocity, yvalue / y_max_velocity, 
        #            alpha=0.6, label='original data', color='blue')
        
        # if fitted_below_data:
        #     # 绘制拟合线
        #     y_fit_range = y_below / y_max_velocity
        #     u_fit_values = (y_below / y_max_velocity)**(1/fitted_below_data['av'])
        #     ax1.plot( u_fit_values, y_fit_range,'r-', linewidth=2, 
        #             label=f'fitting line (av={av_fitted:.3f}, R²={r_squared_below:.3f})')
        
        # # ax1.set_xscale('u/u_max')
        # # ax1.set_yscale('z/Hmax')
        # ax1.set_ylabel('y/y_max')
        # ax1.set_xlabel('u/u_max')
        # ax1.set_title('fittling below y_max ')
        # ax1.legend()
        # ax1.grid(True, alpha=0.3)
        
        # # 子图2: ymax以上拟合（线性坐标）
        # ycoordinate = (yvalue- y_max_velocity) / (np.max(yvalue) - y_max_velocity) 
        # ax2.scatter(uavalue/u_max_velocity, ycoordinate,alpha=0.6, label='original data', color='blue')
        

        # if fitted_above_data and 'fit_results' in fitted_above_data:
        #     X_plot = fitted_above_data['normalized_height']
            
            
        #     # 最终选择的拟合曲线
        #     u_fit_final = np.exp(-fitted_above_data['beta'] * (X_plot ** fitted_above_data['gamma']))
        #     ax2.plot(u_fit_final, X_plot, 'orange', linewidth=3,
        #             label=f'Above fit (β={fitted_above_data["beta"]:.3f}, γ={fitted_above_data["gamma"]:.3f})')
        # # 保存图片
        
        # filename = f"fitting_plot_{time_v}_x{a_id}.png"
        # filepath = os.path.join(plot_dir, filename)
        # ax2.grid(True, alpha=0.3)
        # ax2.legend()
        # ax2.set_xlabel('u/u_max')
        # ax2.set_ylabel('(y - y_max)/(H - y_max)')
        # ax2.set_title('fittling above y_max ')
        # plt.tight_layout()
        # plt.savefig(filepath, dpi=300, bbox_inches='tight')
        # plt.close()
        
        
        # print(f"拟合图已保存至: {filepath}")


        # 创建图形 - 显示组合的无量纲速度剖面 combined fitting plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 分离 below 和 above 区域的数据点
        mask_below = yvalue <= y_max_velocity
        mask_above = yvalue > y_max_velocity
        
        # Below 区域：y/y_max
        y_below_norm = yvalue[mask_below] / y_max_velocity
        u_below_norm = uavalue[mask_below] / u_max_velocity
        
        # Above 区域：(y-y_max)/(H-y_max)，映射到 [1, 2] 区间
        H = np.max(yvalue)
        y_above_norm = 1 + (yvalue[mask_above] - y_max_velocity) / (H - y_max_velocity)
        u_above_norm = uavalue[mask_above] / u_max_velocity
        
        # 绘制原始数据点
        ax.scatter(u_below_norm, y_below_norm, alpha=0.6, 
                label='Below z_max (z/z_max)', color='blue', s=30)
        ax.scatter(u_above_norm, y_above_norm, alpha=0.6, 
                label='Above z_max (1+(z-z_max)/(H-z_max))', color='green', s=30)
        
        # 标记分界点 (y/y_max = 1)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                label='z/z_max = 1 (Max velocity point)')
        
        # 绘制 below y_max 的拟合曲线 - 使用原始数据点
        if fitted_below_data:
            # 直接使用 below 区域的原始 y 值（无量纲化后）
            y_fit_below_norm = y_below_norm
            u_fit_below_norm = y_fit_below_norm ** (1/fitted_below_data['av'])
            
            # 按 y 值排序以便绘制连续曲线
            sort_idx = np.argsort(y_fit_below_norm)
            y_fit_below_norm_sorted = y_fit_below_norm[sort_idx]
            u_fit_below_norm_sorted = u_fit_below_norm[sort_idx]
            
            ax.plot(u_fit_below_norm_sorted, y_fit_below_norm_sorted, 'r-', linewidth=2, 
                    label=f'Below fit (av={fitted_below_data["av"]:.3f}, R²={fitted_below_data["r_squared"]:.3f})')
        
        # 绘制 above y_max 的拟合曲线 - 使用原始数据点
        if fitted_above_data and 'fit_results' in fitted_above_data:
            # 直接使用 above 区域的原始 y 值（无量纲化后）
            y_fit_above_norm = y_above_norm
            
            # 将 [1,2] 映射回 [0,1] 用于拟合公式计算
            normalized_height = y_fit_above_norm - 1.0
            
            # 计算对应的拟合速度
            u_fit_above_norm = np.exp(-fitted_above_data['beta'] * 
                                    (normalized_height ** fitted_above_data['gamma']))
            
            # 按 y 值排序以便绘制连续曲线
            sort_idx = np.argsort(y_fit_above_norm)
            y_fit_above_norm_sorted = y_fit_above_norm[sort_idx]
            u_fit_above_norm_sorted = u_fit_above_norm[sort_idx]
            
            ax.plot(u_fit_above_norm_sorted, y_fit_above_norm_sorted, 'orange', linewidth=2,
                    label=f'Above fit (β={fitted_above_data["beta"]:.3f}, γ={fitted_above_data["gamma"]:.3f},R²={r_squared_above:.3f})')
        
        # 设置坐标轴标签和标题
        ax.set_xlabel('u/u_max')
        ax.set_ylabel('Below: z/z_max | Above: 1 + (z - z_max)/(H - z_max)', fontsize=12)
        ax.set_title(f'Combined Velocity Profile (Time: {time_v})')
        
        # 设置y轴刻度标签，显示实际意义
        y_ticks = [0, 0.5, 1.0, 1.5, 2.0]
        y_tick_labels = ['0', '0.5', '1.0\n(z=z_max)', '0.5\n(z-z_max)/(H-z_max)', '1.0\n(z=H)']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 2.1])
        
        # 保存图片
        filename = f"combined_fitting_plotcombined_{time_v}_x{a_id}.png"
        filepath = os.path.join(plot_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"组合拟合图已保存至: {filepath}")

    return u_max_velocity,beta_fitted,y_max_velocity,gamma_fitted,av_fitted,r_squared_below,r_squared_above

def calculate_derived_values( 
        xvalue,yvalue, alpha, uavalue, save_plot=True,plot_dir="./plots",time_v=0,A=0.5
          ):
    
    yvalue =yvalue
    uavalue =uavalue
    A=A
    time_v=time_v

    u_max_velocity,beta_fitted,y_max_velocity,gamma_fitted,av_fitted,r_squared_below,r_squared_above= fit_velocity_profile(yvalue,uavalue,A,save_plot,plot_dir,time_v)
    
    
    
    """计算衍生量"""

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

def process_time_step(sol, time_v, X, Y,Z,dx,dy,z0,A):
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
        xvalue,yvalue, alpha_value, uavalue, save_plot=True, plot_dir="./plots", time_v=time_v,A=A
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
    sol = "/media/amber/53EA-E81F/PhD/case231020_5"
    # sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case091020_5"
  

    X, Y, Z = fluidfoam.readmesh(sol)
    z0 = 0.135
    # z0 = 0.1
    # 2. 提取 Z=0 的平面（浮点数比较用 np.isclose）
    z_mask = np.isclose(Z, z0)  # 或用 Z == 0（如果数据是精确的）
    X = X[z_mask]
    Y = Y[z_mask]
    dx = np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=0)
    times = np.arange(4, 6, 1)  # 对应原来的1-79,步长2
    results = []
    BASE_PATH = '/home/amber/postpro/selecting_variant/'
    FILE_PREFIX = 'case231020_5middle'  # 修改这里即可自动更新文件名
    # FILE_PREFIX = 'case091020_5middle'  # 修改这里即可自动更新文件名

    
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
            result = process_time_step(sol, time_v, X, Y, Z, dx, dy, z0,A)
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
