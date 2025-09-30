import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import TwoSlopeNorm

### this code is used to calculate the mean velocity and perturbation velocity field of the turbidity current at every time step #####

# 设置路径和输出目录
#sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
#sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
#sol = "/home/amber/OpenFOAM/amber-v2306/Marino/case0704_4"
sol = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_6"
output_dir = "/home/amber/postpro/u_umean_saline"
os.makedirs(output_dir, exist_ok=True)

# 参数设置
alpha_threshold = 1e-4   # alpha.a 的头部阈值
y_min = 0                # 垂向积分下限（避免壁面影响）
 

# 读取时间和网格
#times = fluidfoam.get_time(sol)  # 自动获取所有时间步
#times = sorted([float(t) for t in times if t.replace('.', '').isdigit()])
# 读取网格数据 fludifoam.readmesh(sol) 读取的是网格中心点的数据而非网格节点的数据
X, Y, Z = fluidfoam.readmesh(sol)
xi = np.linspace(X.min(), X.max(), 500)
yi = np.linspace(Y.min(), Y.max(), 3000)
xi, yi = np.meshgrid(xi, yi)
#times = [12,13,14] 
#times = [15,16,17,18,19,20,21,22]
times  = [10]
#times = np.arange(5, 11, 2)

    # 通用参数
FIG_SIZE = (40, 6)
X_LIM = (0, 2.5)
ALPHA_CONTOUR_PARAMS = {
        'levels': [1e-5],
        'colors': 'black',
        'linewidths': 2,
        'linestyles': 'dashed',
        'zorder': 3
    }

def calculate_q_criterion(dUx, dUy, dVx, dVy):
    """
    计算Q准则值
    Q = 0.5*(||Ω||² - ||S||²)
    其中 Ω = 0.5*(∇U - (∇U)^T) 是涡量张量
         S = 0.5*(∇U + (∇U)^T) 是应变率张量
    """


    
    # 计算应变率张量S和涡量张量Ω
    S_xx = dUx
    S_xy = 0.5*(dUy + dVx)
    S_yx = S_xy
    S_yy = dVy
    
    Omega_xy = 0.5*(dVx - dUy)
    Omega_yx = -Omega_xy
    
    # 计算张量的Frobenius范数
    S_norm = S_xx**2 + S_xy**2 + S_yx**2 + S_yy**2
    Omega_norm = Omega_xy**2 + Omega_yx**2
    
    # 计算Q值
    Q = 0.5*(Omega_norm - S_norm)
    
    return Q


def signed(Q, omega_z):
    """保持符号的平滑滤波"""
    # 分离符号与幅值
    sign = np.sign(omega_z)
    magnitude = np.abs(omega_z)
    

    
    # 重建带符号的涡量场
    Q_signed_1 = sign * Q
    
    # 重新与Q准则结合
    return np.where(Q > 0, Q_signed_1, np.nan)

def signed_smooth(Q, omega_z, sigma=1.5):
    """保持符号的平滑滤波"""
    # 分离符号与幅值
    sign = np.sign(omega_z)
    magnitude = np.abs(omega_z)
    
    # 仅对幅值进行高斯滤波
    magnitude_smooth = gaussian_filter(magnitude, sigma=sigma)
    
    # 重建带符号的涡量场
    Q_signed = sign * magnitude_smooth
    
    # 重新与Q准则结合
    return np.where(Q > 0, Q_signed, np.nan)




#for time_v in times:
for i in range(len(times)):
    time_v = times[i]
    # 读取场数据
    Ua_A = fluidfoam.readvector(sol, str(time_v), "U")
    alpha_A = fluidfoam.readscalar(sol, str(time_v), "alpha.saline")
    gradU = fluidfoam.readtensor(sol, str(time_v), "grad(U)")
    vorticity = fluidfoam.readvector(sol, str(time_v), "vorticity")
    gradU_x = gradU[0]  # dUx/dx
    gradU_y = gradU[3]  # dUx/dy
    gradV_x = gradU[1]  # dUy/dx
    gradV_y = gradU[4]  # dUy/dy
    omega_z = vorticity[2]  # 计算二维流场的z方向涡量分量
    
    

    # --- 定位头部位置（alpha.a ≈ alpha_threshold 的最大 x 坐标）---
    head_x = None
    for x in np.unique(X):
        mask = (X == x) & (Y >= y_min) & (alpha_A > alpha_threshold)
        if np.any(mask):
            head_x = x
    if head_x is None:
        print(f"Warning: No head found at t={time_v}")
        continue
    #print( f"Processing time {time_v} with head_x={head_x:.3f}m")
    #print(f"X d position at y={Y[0]} m")
    #print(f"X ***d position at y={Y[1]} m")

    # --- 遍历所有 x < head_x 的坐标 ---
    results = []
    x_coords = np.unique(X[(X <= head_x) & (X >= 0)])  # 仅处理有效 x 范围
    for i, xx in enumerate(x_coords):
        mask = (X == xx) & (alpha_A > 1e-5) & (Y > 0)
        if not np.any(mask):
            continue
        
        ya = Y[mask]
        ua = Ua_A[0][mask]
        alpha = np.maximum(alpha_A[mask], 0)  # 确保 alpha ≥ 0

        # 按 y 排序
        sort_idx = np.argsort(ya)
        ya = ya[sort_idx]
        ua_x = ua[sort_idx]
        alpha_vals = alpha[sort_idx]


        # Find the front height
        # Find the front height with y > 0.005 condition
        y_threshold = 0.005  # 定义y的阈值

        # 找到速度符号变化的点（仅考虑y > y_threshold的区域）
        valid_mask = ya > y_threshold
        sign_changes = np.where(np.diff(np.sign(ua[valid_mask])))[0]
        
        max_ya_crossing_index = sign_changes[np.argmax(ya[sign_changes])] + 1 if len(sign_changes) > 0 else len(ya) - 1
        y_crossing = ya[max_ya_crossing_index]
        u_crossing = ua[max_ya_crossing_index]

        # Vectorized integration
        ua_alpha = ua * alpha
        integral = np.trapz(ua_alpha[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU = np.trapz(ua[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU2 = np.trapz(ua[:max_ya_crossing_index]**2, ya[:max_ya_crossing_index])
        integral2 = np.trapz((ua[:max_ya_crossing_index] * alpha[:max_ya_crossing_index])**2, ya[:max_ya_crossing_index])
        
        # Calculate derived quantities
        U = integralU2 / integralU if integralU != 0 else 0
        H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0
        H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0


        # 首先筛选出 y > 0.005 的点
        y_threshold = 0.005
        valid_mask = ya > y_threshold

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
                y_crossing_alpha = ya[max_ya_crossing_index_alpha]  # 修正变量名
                u_crossing_alpha = ua[max_ya_crossing_index_alpha]
                #print(f"Found y_crossing at {y_crossing_alpha} for xx={xx}")
            else:
                # 如果没有找到，使用有效范围内的最大y值
                max_ya_crossing_index_alpha = np.where(valid_mask)[0][-1]
                y_crossing_alpha = ya[max_ya_crossing_index_alpha]  # 修正变量名
                u_crossing_alpha = ua[max_ya_crossing_index_alpha]
                #print(f"No alpha < {alpha_threshold} found above y={y_threshold} for xx={xx}, using max y={y_crossing_alpha}")
        else:
            # 如果没有y>0.005的点，使用最后一个点
            max_ya_crossing_index_alpha = len(ya) - 1
            y_crossing_alpha = ya[max_ya_crossing_index_alpha]  # 修正变量名
            u_crossing_alpha = ua[max_ya_crossing_index_alpha]  # 无法定义

        # Vectorized integration
        ua_alpha_alpha = ua * alpha
        integral_alpha = np.trapz(ua_alpha_alpha[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha])
        integralU_alpha = np.trapz(ua[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha])
        integralU2_alpha = np.trapz(ua[:max_ya_crossing_index_alpha]**2, ya[:max_ya_crossing_index_alpha])
        integral2_alpha = np.trapz((ua[:max_ya_crossing_index_alpha] * alpha[:max_ya_crossing_index_alpha])**2, ya[:max_ya_crossing_index_alpha])
        
        # Calculate derived quantities
        U_alpha = integralU2_alpha / integralU_alpha if integralU_alpha != 0 else 0
        H_alpha = integral_alpha**2 / integral2_alpha if integral2_alpha != 0 else 0
        ALPHA_alpha = integral_alpha / integralU_alpha if integralU_alpha != 0 else 0
        H_depth_alpha = integralU_alpha**2 / integralU2_alpha if integralU2_alpha != 0 else 0




        results.append({
            "Time": time_v,
            "x": xx,
            "U": U,
            "H": H,
            "ALPHA": ALPHA,
            "y_crossing": y_crossing,
            "U_alpha"   : U_alpha,
            "H_alpha"   : H_alpha,
            "ALPHA_alpha": ALPHA_alpha,
            "y_crossing_alpha": y_crossing_alpha
        })


    # 保存当前时间步的结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"integration_results_t{time_v}.csv"), index=False)
    print(f"Results saved for t={time_v} (head_x={head_x:.2f}m)")

    
    
    # --- 计算速度扰动 U_a' = U_a - U(x) ---
    # 从DataFrame中获取U值，并创建x到U的映射
    x_U_mapping = dict(zip(df['x'], df['U']))
    x_U_mapping_alpha = dict(zip(df['x'], df['U_alpha']))
    
    #U_perturb = np.zeros_like(Ua_A[0])
    U_perturb = Ua_A[0].copy()  # 直接使用 Ua_A[0] 的形状
    for x in x_coords:
        if x in x_U_mapping:  # 确保x在计算结果中
            mask = (X == x)
            U_perturb[mask] = Ua_A[0][mask] - x_U_mapping[x]

    U_perturb_alpha = Ua_A[0].copy()  # 直接使用 Ua_A[0] 的形状
    for x in x_coords:
        if x in x_U_mapping_alpha:  # 确保x在计算结果中
            mask = (X == x)
            U_perturb_alpha[mask] = Ua_A[0][mask] - x_U_mapping_alpha[x]

    # --- 插值到规则网格（可选，使绘图更平滑）---
    
    U_perturb_i = griddata((X, Y), U_perturb, (xi, yi), method='cubic')
    U_perturb_i_alpha = griddata((X, Y), U_perturb_alpha, (xi, yi), method='cubic')
    # 插值原始速度场（用于流线方向）
    uxi = griddata((X, Y), Ua_A[0], (xi, yi), method='linear')  # Ux 分量
    uyi = griddata((X, Y), Ua_A[1], (xi, yi), method='linear')  # Uy 分量
    alpha_i = griddata((X, Y), alpha_A, (xi, yi), method='linear')
    vorticity_i = griddata((X, Y), omega_z, (xi, yi), method='linear')
    

    # 计算Q准则
    Q = calculate_q_criterion(gradU_x, gradU_y, gradV_x, gradV_y)
    Q_i = griddata((X, Y), Q, (xi, yi), method='linear')

    Q_signed = signed(Q, omega_z)
    Q_signed_i = griddata((X, Y), Q_signed, (xi, yi), method='linear')

    Q_signed_smoothed = signed_smooth(Q, omega_z, sigma=1.5)
    Q_signed_smoothed_i = griddata((X, Y), Q_signed_smoothed, (xi, yi), method='linear')
    


    # 全局绘图参数设置
    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 28,
        'axes.labelsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })


    H = 0.3
    positions = {
        '1/6H': head_x - (1/6)*H,
        '1/4H': head_x - 0.25*H,
        '1/2H': head_x - 0.5*H,
        'H': head_x - H
    }
    
    y_text = 0.32
    
    

# #def plot_velocity_vectors(X, Y, U_perturb, Ua_A, xi, yi, alpha_i, time_v, output_dir):
    
#     """绘制速度矢量图"""
#     plt.figure(figsize=FIG_SIZE)
    
#     # 矢量场参数
#     skip = 5
#     scale = 20
#     width = 0.002
    
#     plt.quiver(
#         X[::skip], Y[::skip],
#         U_perturb[::skip], Ua_A[1][::skip],
#         scale=scale,
#         width=width,
#         color='blue'
#     )
    
#     if 'alpha_i' in locals():
#         plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

#     for label, x_pos in positions.items():
#         plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
#         plt.text(x_pos + 0.01, y_text, f'{label}', 
#                 fontsize=15, zorder=3, color='b')
#         #y_text += 0.04    
    
#     plt.xlabel('x [m]')
#     plt.ylabel('y [m]')
#     plt.xlim(*X_LIM)
#     plt.title(f'Velocity Vectors (t={time_v}) - No Interpolation')
#     plt.savefig(os.path.join(output_dir, f'velocity_vectors_t{time_v}.png'), 
#               dpi=300, bbox_inches='tight')
#     plt.close()


    """绘制扰动速度流线图"""
    plt.figure(figsize=FIG_SIZE)
    
    # 归一化和缩放
    magnitude = np.sqrt(U_perturb_i**2 + uyi**2)
    U_perturb_i_norm = np.where(magnitude > 0, U_perturb_i / magnitude, 0)
    U_perturb_i_scaled = np.clip(U_perturb_i, -0.02, 0.02)
    
    strm = plt.streamplot(
        xi, yi, U_perturb_i, uyi,
        color=U_perturb_i_scaled,
        cmap=plt.cm.rainbow,
        linewidth=1,
        density=8,
        arrowsize=3,
        arrowstyle='->',
        zorder=1
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04
    
    cbar = plt.colorbar(strm.lines, label="Velocity Perturbation $U_a\'$ [m/s]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Perturbation Velocity Streamlines (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'perturbation_streamlines_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()

    """绘制扰动速度流线图alpha"""
    plt.figure(figsize=FIG_SIZE)
    
    # 归一化和缩放
    magnitude = np.sqrt(U_perturb_i_alpha**2 + uyi**2)
    U_perturb_i_norm = np.where(magnitude > 0, U_perturb_i_alpha / magnitude, 0)
    U_perturb_i_alphascaled = np.clip(U_perturb_i_alpha, -0.02, 0.02)
    
    strm = plt.streamplot(
        xi, yi, U_perturb_i_alpha, uyi,
        color=U_perturb_i_alphascaled,
        cmap=plt.cm.rainbow,
        linewidth=1,
        density=8,
        arrowsize=3,
        arrowstyle='->',
        zorder=1
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04
    
    cbar = plt.colorbar(strm.lines, label="Velocity Perturbation $U_aaalpha\'$ [m/s]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Perturbation Velocity Streamlines alpha (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'perturbation_streamlines_alpha_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()


    """绘制原始速度流线图"""
    plt.figure(figsize=FIG_SIZE)
    uxi_scaled = np.clip(uxi, -0.12, 0.12)
    
    strm = plt.streamplot(
        xi, yi, uxi, uyi,
        color=uxi_scaled,
        cmap=plt.cm.rainbow,
        linewidth=1,
        density=5,
        arrowsize=2,
        arrowstyle='->',
        zorder=2
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04
    
    cbar = plt.colorbar(strm.lines, 
                       label="Velocity streamline $U_a$ [m/s]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Original Velocity Streamlines (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'origin_streamlines_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()


    """omega_z"""
    plt.figure(figsize=FIG_SIZE)
    
    V_levels = np.linspace(-10.5, 10.5, 81)
    V_contour = plt.contourf(
        xi, yi, vorticity_i,
        levels=V_levels,
        cmap='bwr',
        extend='both'
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)
    

    
    
    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04
    
    cbar = plt.colorbar(V_contour, label='$\omega_z$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Vorticity (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'Vorticity_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()



    # """绘制优化后的Q准则涡分布图"""
    

    # plt.figure(figsize=FIG_SIZE)
    
    # Q_levels = np.linspace(-1.5, 1.5, 11)
    # Q_contour_signed = plt.contourf(
    #     xi, yi, Q_signed_i,
    #     levels=Q_levels,
    #     cmap='bwr',
    #     extend='both'
    # )
    
    # plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    # for label, x_pos in positions.items():
    #     plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
    #     plt.text(x_pos + 0.01, y_text, f'{label}', 
    #             fontsize=15, zorder=3, color='b')
    #     #y_text += 0.04
    
    # cbar = plt.colorbar(Q_contour_signed, label='Q-Criterion')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.xlim(*X_LIM)
    # plt.title(f'Q-Criterion Contours_signed (t={time_v})')
    # plt.savefig(os.path.join(output_dir, f'Q_criterion_t{time_v}_signed.png'), 
    #           dpi=300, bbox_inches='tight')
    # plt.close()

    # """绘制考虑涡量后的Q准则涡分布图"""
    

    # plt.figure(figsize=FIG_SIZE)
    
    # Q_levels = np.linspace(-5, 5, 41)
    # Q_contour_signed = plt.contourf(
    #     xi, yi, Q_signed_smoothed_i,
    #     levels=Q_levels,
    #     cmap='bwr',
    #     extend='both'
    # )
    
    # plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    # for label, x_pos in positions.items():
    #     plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
    #     plt.text(x_pos + 0.01, y_text, f'{label}', 
    #             fontsize=15, zorder=3, color='b')
    #     #y_text += 0.04
    
    # cbar = plt.colorbar(Q_contour_signed, label='Q-Criterion')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.xlim(*X_LIM)
    # plt.title(f'Vortex indentification via Q_criterion with vorticity magnitude (t={time_v})')
    # plt.savefig(os.path.join(output_dir, f'Q_criterion_t{time_v}_signed_omega.png'), 
    #           dpi=300, bbox_inches='tight')
    # plt.close()


print(f"All results saved to: {output_dir}")



#print(f"All results saved to: {output_dir}")
