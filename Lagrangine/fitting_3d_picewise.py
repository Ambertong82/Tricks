import numpy as np
import pandas as pd
import os
import fluidfoam
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
### this code is used to extract the data at specific x location (head-0.3) at every time step #####

# 常量定义

y_min = 0
alpha_threshold = 1e-5
slope_threshold = 1  # BELOW区域斜率变化阈值 0.75 0.25



def get_a_identifier(a_value):
    frac = Fraction(a_value).limit_denominator()
    return f"{frac.numerator}{frac.denominator}"

def get_output_files():
    a_id = get_a_identifier(A) # 例如A=10/12→"1012"

    return {
        # 'yy': f'x{a_id}xyy.csv',
        'ua': f'x{a_id}{slope_threshold}xua.csv',
        # 'ub': f'x{a_id}xub.csv',
        'U': f'x{a_id}{slope_threshold}xU.csv',
        'H_depth': f'x{a_id}{slope_threshold}xH_depth.csv',
        'ycrossing': f'x{a_id}{slope_threshold}xycrossing.csv',
        'Udir': f'x{a_id}{slope_threshold}xUdir.csv',
        'u_max_velocity': f'x{a_id}{slope_threshold}xu_max_velocity.csv',
        'y_max_velocity': f'x{a_id}{slope_threshold}xy_max_velocity.csv',
        # 新增拟合结果文件
        'av_fitted': f'x{a_id}{slope_threshold}xav_fitted.csv',
        'beta_fitted': f'x{a_id}{slope_threshold}xbeta_fitted.csv',
        'gamma_fitted': f'x{a_id}{slope_threshold}xgamma_fitted.csv',
        'r_squared_below': f'x{a_id}{slope_threshold}xr_squared_below.csv',
        'r_squared_above': f'x{a_id}{slope_threshold}xr_squared_above.csv',
        'y_boundary': f'x{a_id}{slope_threshold}xy_boundary.csv',
        'y_above_boundary': f'x{a_id}{slope_threshold}xy_above_boundary.csv',
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

    # y_connect1 = 0.001
    # y_connect2 = y_max_velocity
    # u_connect1 = np.interpolate(y_connect1, yvalue, uavalue)
    # u_connect2 = u_max_velocity


    
   # ==================== 拟合1: ymax以下位置 ====================
    # 初始化变量
    fitted_below_data = None
    fitted_above_data = None

    
    # ==================== 区域划分和拟合 ====================
    # 定义分界线
    

    # 区域划分
   
    mask_below = (yvalue <= y_max_velocity)
    mask_above = (yvalue > y_max_velocity )

    y_below = yvalue[mask_below]
    u_below = uavalue[mask_below]
    y_above = yvalue[mask_above]
    u_above = uavalue[mask_above]

    # 初始化结果变量
    fitted_below_data = None
    fitted_above_data = None

    av_fitted = 0.0
    beta_fitted = 0.0
    gamma_fitted = 1.0
    r_squared_below = 0.0
    r_squared_above = 0.0
   
# ==================== 1. BELOW区域分界点检测 ====================
    if len(y_below) > 2 and y_max_velocity > 0 and u_max_velocity > 0:
        try:
            # 找到速度最大值对应的索引
            max_u_index = np.argmax(uavalue)
            
            # 步骤1: 用ymax附近的4-5个点拟合参考曲线
            start_index_ref = max(0, max_u_index - 7)  # 从ymax往上数第8个点开始
            end_index_ref = max_u_index  # 到ymax
            
            y_ref = yvalue[start_index_ref:end_index_ref+1]
            u_ref = uavalue[start_index_ref:end_index_ref+1]
            
            # 归一化并取对数
            y_ref_norm = y_ref / y_max_velocity
            u_ref_norm = u_ref / u_max_velocity
            
            X_ref = np.log(y_ref_norm)
            Y_ref = np.log(u_ref_norm)
            
            # 参考斜率（幂指数倒数）
            slope_ref = np.sum(X_ref * Y_ref) / np.sum(X_ref**2)
            av_ref = 1.0 / slope_ref if abs(slope_ref) > 1e-6 else 1.0
            
            print(f"参考拟合(点{start_index_ref}-{end_index_ref}): av_ref = {av_ref:.4f}")
            
            # 步骤2: 从第6个点开始，检查局部斜率变化
            boundary_index = None
            min_points_for_slope = 3  # 计算局部斜率所需的最小点数
            
            # 从ymax-6开始向前检查
            for test_index in range(max_u_index - 8, 0, -1):
                if test_index < min_points_for_slope:
                    break
                    
                # 方法1: 计算当前点附近的局部斜率（数值微分）
                # 取当前点及后2个点，共3个点计算斜率
                idx_start = max(0, test_index - 2)
                idx_end = min(len(yvalue) - 1, test_index )
                
                if idx_end - idx_start < 2:  # 至少需要3个点
                    continue
                    
                y_local = yvalue[idx_start:idx_end+1]
                u_local = uavalue[idx_start:idx_end+1]
                
                # 计算局部斜率（对数空间）
                y_local_norm = y_local / y_max_velocity
                u_local_norm = u_local / u_max_velocity
                
                # 数值微分计算局部斜率
                dy = np.diff(np.log(y_local_norm))
                du = np.diff(np.log(u_local_norm))
                local_slopes = du / dy
                
                # 取平均局部斜率
                local_slope = np.mean(local_slopes) if len(local_slopes) > 0 else slope_ref
                
                # 方法2: 或者用当前点与前2个点的斜率
                # if test_index >= 2:
                #     y_window = yvalue[test_index-2:test_index+1]
                #     u_window = uavalue[test_index-2:test_index+1]
                #     y_window_norm = y_window / y_max_velocity
                #     u_window_norm = u_window / u_max_velocity
                    
                #     X_window = np.log(y_window_norm)
                #     Y_window = np.log(u_window_norm)
                #     window_slope = np.sum(X_window * Y_window) / np.sum(X_window**2)
                # else:
                #     window_slope = slope_ref
                
                # 综合两种方法的斜率估计
                combined_slope =   local_slope

                if combined_slope < 0:
                    boundary_index = test_index
                    print(f"找到分界点(斜率负值): 索引={boundary_index}, y={yvalue[boundary_index]:.4f}")
                    break
                
                # 计算斜率差异
                # slope_diff = (abs((combined_slope) - abs(slope_ref)) / abs(slope_ref))
                slope_diff = abs((combined_slope - slope_ref)/slope_ref)
                
                print(f"检查点{test_index}: y={yvalue[test_index]:.4f}, 斜率差异={slope_diff:.4f}")
                # print(f"combined_slope: {combined_slope}, slope_ref: {slope_ref}")
                
                # 如果斜率差异超过阈值，认为找到分界点
                if slope_diff > slope_threshold:  # 阈值可调整
                    boundary_index = test_index
                    print(f"找到分界点: 索引={boundary_index}, y={yvalue[boundary_index]:.4f}")
                    break
            
            # 步骤3: 根据找到的分界点进行最终拟合
            if boundary_index is not None:
                # 从分界点到ymax进行拟合
                y_final = yvalue[boundary_index:max_u_index+1]
                u_final = uavalue[boundary_index:max_u_index+1]
            else:
                # 没找到明显分界点，使用所有below数据
                y_final = y_below
                u_final = u_below
                boundary_index = 0
                print("未找到明显分界点，使用全部BELOW数据")
            
            # 最终拟合
            y_final_norm = y_final / y_max_velocity
            u_final_norm = u_final / u_max_velocity
            
            X_final = np.log(y_final_norm)
            Y_final = np.log(u_final_norm)
            
            slope_final = np.sum(X_final * Y_final) / np.sum(X_final**2)
            av_final = 1.0 / slope_final if abs(slope_final) > 1e-6 else 1.0
            
            # 计算R²
            y_pred = slope_final * X_final
            ss_res = np.sum((Y_final - y_pred) ** 2)
            ss_tot_uncen = np.sum(Y_final ** 2)
            r_squared = 1 - (ss_res / ss_tot_uncen) if ss_tot_uncen != 0 else 0
            
            print(f"BELOW区域最终拟合: av = {av_final:.4f}, R² = {r_squared:.4f}")
            
            # 保存结果
            fitted_below_data = {
                'y_original': y_final,
                'u_original': u_final,
                'y_normalized': y_final_norm,
                'u_normalized': u_final_norm,
                'av': av_final,
                'r_squared_below': r_squared,
                'boundary_index': boundary_index,
                'boundary_y': yvalue[boundary_index],
                'region': 'below_auto_boundary'
            }
            av_fitted = av_final
            
        except Exception as e:
            print(f"BELOW区域分界点检测失败: {e}")
            fitted_below_data = None
       


    # ==================== 3. ABOVE区域拟合（保持不变） ====================

    beta_fitted = 0.0
    gamma_fitted = 1.0
    r_squared_above = 0.0
    fitted_above_data = None

    # 存储两种方法的结果
    fit_results = {'original': None, 'log': None}

    # ==================== 2. ABOVE区域分界点检测（窗口回归法） ====================
    if len(y_above) > 2 and y_max_velocity > 0 and u_max_velocity > 0:
        try:
            # 找到速度最大值对应的索引（在原始数据中）
            max_u_index_global = np.argmax(uavalue)
            H = np.max(yvalue)
            H_minus_Hm = H - y_max_velocity
            
            if H_minus_Hm <= 0:
                print("ABOVE区域高度无效，跳过")
                fitted_above_data = None
            else:
                # 步骤1: 用ymax往上5个点进行参考拟合
                num_ref_points = 5
                start_index_ref = max_u_index_global + 1
                end_index_ref = min(len(yvalue) - 1, max_u_index_global + num_ref_points)
                
                if end_index_ref <= start_index_ref:
                    print("ABOVE区域数据点不足，跳过")
                    fitted_above_data = None
                else:
                    y_ref_above = yvalue[start_index_ref:end_index_ref+1]
                    u_ref_above = uavalue[start_index_ref:end_index_ref+1]
                    
                    print(f"ABOVE参考点范围: 索引{start_index_ref}到{end_index_ref}, 共{len(y_ref_above)}个点")
                    
                    # ABOVE区域归一化
                    y_ref_norm_above = (y_ref_above - y_max_velocity) / H_minus_Hm
                    u_ref_norm_above = u_ref_above / u_max_velocity
                    
                    # 确保数据有效性
                    valid_mask = (y_ref_norm_above > 0) & (y_ref_norm_above <= 1.0) & \
                            (u_ref_norm_above >= 0) & (u_ref_norm_above <= 1.0)
                    y_ref_valid = y_ref_norm_above[valid_mask]
                    u_ref_valid = u_ref_norm_above[valid_mask]
                    
                    if len(y_ref_valid) >= 3:
                        # ABOVE区域参考拟合（指数衰减形式）
                        def above_fit_func(norm_h, beta, gamma):
                            return np.exp(-beta * (norm_h ** gamma))
                        
                        try:
                            p0 = [2.0, 2.0]
                            bounds = ([0.1, 0.1], [20.0, 10.0])
                            
                            popt_ref, _ = curve_fit(above_fit_func, y_ref_valid, u_ref_valid, 
                                                p0=p0, maxfev=5000, bounds=bounds)
                            beta_ref, gamma_ref = popt_ref
                            
                            print(f"ABOVE参考拟合: beta_ref = {beta_ref:.4f}, gamma_ref = {gamma_ref:.4f}")
                            
                            # 步骤2: 从第6个点开始使用窗口回归法检测分界点
                            boundary_index_above = None
                            
                            
                            # 从max_u_index_global+6开始向后检查（向水面方向）
                            for test_index in range(max_u_index_global + 6, len(yvalue)):
                                # 方法1: 计算当前点附近的局部梯度（数值微分）
                                idx_start = max(max_u_index_global + 1, test_index - 2)
                                idx_end = min(len(yvalue) - 1, test_index)
                                
                                if idx_end - idx_start < 2:  # 至少需要3个点
                                    continue
                                    
                                y_local = yvalue[idx_start:idx_end+1]
                                u_local = uavalue[idx_start:idx_end+1]
                                
                                # ABOVE区域归一化
                                y_local_norm = (y_local - y_max_velocity) / H_minus_Hm
                                u_local_norm = u_local / u_max_velocity
                                
                                # 数值微分计算局部梯度
                                valid_mask_local = (y_local_norm > 0) & (y_local_norm <= 1.0) & \
                                                (u_local_norm >= 0) & (u_local_norm <= 1.0)
                                y_local_valid = y_local_norm[valid_mask_local]
                                u_local_valid = u_local_norm[valid_mask_local]
                                
                                if len(y_local_valid) >= 2:
                                    # 计算瞬时梯度（避免log(0)，使用原始差分）
                                    y_diff = np.diff(y_local_valid)
                                    u_diff = np.diff(u_local_valid)
                                    local_gradients = u_diff / y_diff
                                    local_gradient = np.mean(local_gradients) if len(local_gradients) > 0 else 0
                                else:
                                    local_gradient = 0
                                
                                
                                # 计算当前点的参考梯度（解析解）
                                y_current_norm = (yvalue[test_index] - y_max_velocity) / H_minus_Hm
                                if y_current_norm > 0 and y_current_norm <= 1.0:
                                    # 参考曲线梯度公式
                                    ref_gradient = -beta_ref * gamma_ref * (y_current_norm ** (gamma_ref - 1)) * \
                                                np.exp(-beta_ref * (y_current_norm ** gamma_ref))
                                    
                                    # 计算梯度差异（数值微分梯度 vs 参考梯度）
                                    if ref_gradient != 0:
                                        gradient_diff = abs((local_gradient - ref_gradient) / ref_gradient)
                                    else:
                                        gradient_diff = abs(local_gradient)

                                    # print(f"local_gradient: {local_gradient}, ref_gradient: {ref_gradient}, gradient_diff: {gradient_diff}")    
                                    
                                    # 综合差异（结合梯度差异和beta参数差异）
                                    combined_diff = gradient_diff 
                                    # combined_diff = window_beta_diff
                                    # print(f"ABOVE检查点{test_index}: y={yvalue[test_index]:.4f} ")
                                    
                                    # 如果综合差异超过阈值，认为找到分界点
                                    if combined_diff > slope_threshold:
                                        boundary_index_above = test_index
                                        print(f"ABOVE找到分界点: 索引={boundary_index_above}, y={yvalue[boundary_index_above]:.4f}")
                                        break
                            
                            # 步骤3: 根据分界点进行最终拟合
                            if boundary_index_above is not None:
                                y_final_above = yvalue[max_u_index_global+1:boundary_index_above+1]
                                u_final_above = uavalue[max_u_index_global+1:boundary_index_above+1]
                            else:
                                y_final_above = y_above
                                u_final_above = u_above
                                boundary_index_above = len(yvalue) - 1
                                print("ABOVE未找到明显分界点，使用全部数据")
                            
                            # ABOVE区域最终拟合
                            y_final_norm_above = (y_final_above - y_max_velocity) / H_minus_Hm
                            u_final_norm_above = u_final_above / u_max_velocity
                            
                            valid_mask_final = (y_final_norm_above > 0) & (y_final_norm_above <= 1.0) & \
                                            (u_final_norm_above >= 0) & (u_final_norm_above <= 1.0)
                            X_final_above = y_final_norm_above[valid_mask_final]
                            Y_final_above = u_final_norm_above[valid_mask_final]
                            
                            if len(X_final_above) >= 3:
                                # 最终拟合
                                popt_final, pcov_final = curve_fit(above_fit_func, X_final_above, Y_final_above,
                                                                p0=[beta_ref, gamma_ref], maxfev=5000,
                                                                bounds=([0.1, 0.1], [20.0, 10.0]))
                                beta_final, gamma_final = popt_final
                                
                                # 计算R²
                                y_pred = above_fit_func(X_final_above, beta_final, gamma_final)
                                ss_res = np.sum((Y_final_above - y_pred) ** 2)
                                ss_tot = np.sum((Y_final_above - np.mean(Y_final_above)) ** 2)
                                r_squared_above = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                                
                                print(f"ABOVE最终拟合: beta={beta_final:.4f}, gamma={gamma_final:.4f}, R²={r_squared:.4f}")
                                
                                fitted_above_data = {
                                    'normalized_height': X_final_above,
                                    'u_normalized': Y_final_above,
                                    'y_original': y_final_above[valid_mask_final],
                                    'u_original': u_final_above[valid_mask_final],
                                    'beta': beta_final,
                                    'gamma': gamma_final,
                                    'r_squared_above': r_squared_above,
                                    'boundary_index': boundary_index_above,
                                    'boundary_y': yvalue[boundary_index_above],
                                    'region': 'above_auto_boundary'
                                }
                                beta_fitted = beta_final
                                gamma_fitted = gamma_final
                                r_squared_above = r_squared
                            else:
                                fitted_above_data = None
                                print("ABOVE最终拟合数据不足")
                                
                        except Exception as e:
                            print(f"ABOVE参考拟合失败: {e}")
                            fitted_above_data = None
                    else:
                        fitted_above_data = None
                        print("ABOVE参考数据点不足")
                        
        except Exception as e:
            print(f"ABOVE区域分界点检测失败: {e}")
            fitted_above_data = None

    # ==================== 绘图部分（需要相应修改） ====================
    if save_plot:
        a_id = get_a_identifier(A) # 例如A=10/12→"1012"
        # 修改绘图代码来显示三个区域
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
       
        
         # Below 区域：y/y_max
        
        y_below_norm = y_below / y_max_velocity
        u_below_norm = u_below / u_max_velocity

        

        # Above 区域：(y-y_max)/(H-y_max)，映射到 [1, 2] 区间
        H = np.max(yvalue)
        y_above_norm = 1 + (y_above - y_max_velocity) / (H - y_max_velocity)
        u_above_norm = u_above / u_max_velocity
        
        # 绘制原始数据点
        ax.scatter(u_below_norm, y_below_norm, alpha=0.6, 
                label='Below z_max (z/z_t)', color='blue', s=30)
        
        
        
        ax.scatter(u_above_norm, y_above_norm, alpha=0.6, 
                label='Above z_max (1+(z-z_max)/(H-z_max))', color='green', s=30)
        
        # 标记分界点 (y/y_max = 1)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                label='z/z_max = 1 (Max velocity point)')
        
        ax.axhline(y =fitted_below_data['boundary_y']/y_max_velocity if fitted_below_data else 1.0, color='purple', linestyle='--', alpha=0.7,
                   label= (f"Hc_below = {fitted_below_data['boundary_y']:.3f}" if fitted_below_data else "N/A"))
        
        ax.axhline(y =1 + (fitted_above_data['boundary_y'] - y_max_velocity)/(H - y_max_velocity) if fitted_above_data else 1.0, color='orange', linestyle='--', alpha=0.7,
                     label= (f"Hc_above = {fitted_above_data['boundary_y']:.3f}" if fitted_above_data else "N/A"))
        
        # 绘制拟合曲线
        if fitted_below_data:
            y_fit = fitted_below_data['y_normalized']
            u_fit = y_fit ** (1/fitted_below_data['av']) 
            
            ax.plot(u_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Below fit (av={fitted_below_data["av"]:.3f}, R²={fitted_below_data["r_squared_below"]:.3f})')
        
        
        
        if fitted_above_data:
            y_fit = fitted_above_data['normalized_height']
            u_fit = np.exp(-fitted_above_data['beta'] * (y_fit ** fitted_above_data['gamma']))
            ax.plot(u_fit, 1 + y_fit, 'orange', linewidth=2,
                    label=f'Above fit (β={fitted_above_data["beta"]:.3f}, γ={fitted_above_data["gamma"]:.3f}, R²={fitted_above_data["r_squared_above"]:.3f})')

    
        
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
        filename = f"combined_fitting_plotcombined_{time_v}_x{a_id}_{slope_threshold}.png"
        filepath = os.path.join(plot_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"组合拟合图已保存至: {filepath}")

    return u_max_velocity,beta_fitted,y_max_velocity,gamma_fitted,av_fitted,yvalue[boundary_index],yvalue[boundary_index_above]

def calculate_derived_values( 
        xvalue,yvalue, alpha, uavalue, save_plot=True,plot_dir="./plots",time_v=0,A=0.5
          ):
    
    yvalue =yvalue
    uavalue =uavalue
    A=A
    time_v=time_v

    u_max_velocity,beta_fitted,y_max_velocity,gamma_fitted,av_fitted,yc,ycbove = fit_velocity_profile(yvalue,uavalue,A,save_plot,plot_dir,time_v)
    
    
    
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
        
        'y_boundary': [yc],
        'y_above_boundary': [ycbove],
    }

def process_time_step(sol, time_v, X, Y, Z, dx, dy, z0, A, PLOT_DIR):
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
        xvalue,yvalue, alpha_value, uavalue, save_plot=True, plot_dir=PLOT_DIR, time_v=time_v,A=A,
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
def save_combined_heights(data_dict, output_dir, FILE_PREFIX):
    """专门保存合并的高度数据"""
    a_id = get_a_identifier(A)
    filename = f"x{a_id}{slope_threshold}xy_heights.csv"
    
    data = []
    for item in data_dict:
        row = [
            item['time'],           # 时间
            item['x'],              # x坐标
            item['y_max_velocity'][0],     # y_max_velocity
            item['y_boundary'][0],         # y_boundary (BELOW区域分界点)
            item['y_above_boundary'][0],   # y_above_boundary (ABOVE区域分界点)
            item['ycrossing'][0]           # y_crossing (零速度交叉点)
        ]
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=['time', 'x', 'y_max_velocity', 'y_boundary', 'y_above_boundary', 'ycrossing'])
    
    # 保存文件
    filepath = os.path.join(output_dir, f"{FILE_PREFIX}_{filename}")
    df.to_csv(filepath, index=False)
    print(f"合并高度数据已保存至: {filepath}")

def save_data(data_dict, output_dir, FILE_PREFIX):
    """保存数据到CSV文件"""
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
        
        # 保存文件（无表头）- 使用传入的输出目录
        filepath = os.path.join(output_dir, f"{FILE_PREFIX}_{filename}")
        df.to_csv(filepath, index=False, header=False)

def main():
    ############### 主程序 ##################
    # 在main函数顶部统一定义所有路径
    sol = "/media/amber/53EA-E81F/PhD/case231020_5"
    
    # 定义所有路径
    # BASE_PATH = '/home/amber/postpro/selecting_variant/'  # 原始路径（可根据需要保留）
    PLOT_DIR = "/home/amber/postpro/fitting_coarse_tc3dmiddle"  # 图片保存路径
    OUTPUT_DIR = "/home/amber/postpro/fitting_results"  # 数据文件保存路径
    FILE_PREFIX = 'case231020_5middle'  # 文件名前缀

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    X, Y, Z = fluidfoam.readmesh(sol)
    z0 = 0.135
    
    # 提取 Z=z0 的平面
    z_mask = np.isclose(Z, z0)
    X = X[z_mask]
    Y = Y[z_mask]
    dx = np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=0)
    
    times = np.arange(4, 11, 1)  # 对应原来的1-79,步长2
    
    # === 定义A值的列表 ===
    a_values = [Fraction(1,4), Fraction(1,3), Fraction(1, 2), Fraction(1, 1)]
    # a_values = [Fraction(1,1)]
    
    # === 对每个A值进行循环 ===
    for a_val in a_values:
        global A
        A = float(a_val)  # 转换为浮点数用于计算
        
        print(f"\n=== 处理 A = {a_val} ===")
        
        # 为当前A值创建新的结果列表
        results = []
        
        for time_v in times:
            result = process_time_step(sol, time_v, X, Y, Z, dx, dy, z0, A, PLOT_DIR)
            if result:
                results.append(result)
        
        if results:
            save_data(results, OUTPUT_DIR, FILE_PREFIX)
            save_combined_heights(results, OUTPUT_DIR, FILE_PREFIX)
            print(f"A = {a_val} 数据处理完成，结果已保存到目录: {OUTPUT_DIR}")
            for name in get_output_files().values():
                print(f"  - {FILE_PREFIX}_{name}")
        else:
            print(f"A = {a_val} 未找到有效数据")

if __name__ == '__main__':
    main()