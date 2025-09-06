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
sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
#sol = "/home/amber/OpenFOAM/amber-v2306/Marino/case0704_4"
output_dir = "/home/amber/postpro/u_umean_tc_d9"
os.makedirs(output_dir, exist_ok=True)

# 参数设置
alpha_threshold = 1e-5   # alpha.a 的头部阈值
y_min = 0                # 垂向积分下限（避免壁面影响）

# 读取时间和网格
#times = fluidfoam.get_time(sol)  # 自动获取所有时间步
#times = sorted([float(t) for t in times if t.replace('.', '').isdigit()])
# 读取网格数据 fludifoam.readmesh(sol) 读取的是网格中心点的数据而非网格节点的数据
X, Y, Z = fluidfoam.readmesh(sol)
#times = [5,6,7,8,9,10,11,12] 
#times = [15,16,17,18,19]
times  = [12.0]
#times = np.arange(7.0, 11.0, 1)

 # 通用参数
FIG_SIZE = (40, 6)
X_LIM = (0.0, 1.5)
ALPHA_CONTOUR_PARAMS = {
        'levels': [1e-4],
        'colors': 'black',
        'linewidths': 2,
        'linestyles': 'dashed',
        'zorder': 3
    }
Height = 0.3


    

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
    Ua_A = fluidfoam.readvector(sol, str(time_v), "U.a")
    alpha_A = fluidfoam.readscalar(sol, str(time_v), "alpha.a")
    gradU = fluidfoam.readtensor(sol, str(time_v), "grad(U.a)")
    gradU_x = gradU[0]  # dUx/dx
    gradU_y = gradU[3]  # dUx/dy
    gradV_x = gradU[1]  # dUy/dx
    gradV_y = gradU[4]  # dUy/dy
    omega_z =  (gradV_x - gradU_y)  # 计算二维流场的z方向涡量分量

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

    for x in x_coords:
        # 提取当前 x 处的垂向数据
        mask = (X == x) & (Y >= y_min)
        ya = Y[mask]
        ua = Ua_A[0][mask]
        alpha = np.maximum(alpha_A[mask], 0)  # 确保 alpha ≥ 0

        # 按 y 排序
        sort_idx = np.argsort(ya)
        ya = ya[sort_idx]
        ua = ua[sort_idx]
        alpha = alpha[sort_idx]

        # 计算 dy（垂向间距）
        dy = np.diff(ya, prepend=ya[0] - 0)  # 初始 dy 假设为第一个点的 y 坐标

        # --- 垂向积分 ---
        # 有效区域：alpha > threshold 且 ua > 0
        valid_mask =  (ua > 0) & (alpha > alpha_threshold) 
        if not np.any(valid_mask):
            continue  # 跳过无效点

        # 积分上限：有效区域的最高 y
        max_ya = ya[valid_mask][-1]
        integral_mask = (ya <= max_ya)

        # 梯形法积分
        ua_alpha = ua[integral_mask] * alpha[integral_mask]
        dy_integral = dy[integral_mask]
        #print(f"Processing x={x:.3f}m at t={time_v} with max_ya={max_ya:.3f}m")

        # 质量通量 ∫(u * alpha) dy
        integral = np.sum(0.5 * (ua_alpha[1:] + ua_alpha[:-1]) * dy_integral[1:])
        # 速度通量 ∫u dy
        integralU = np.sum(0.5 * (ua[integral_mask][1:] + ua[integral_mask][:-1]) * dy_integral[1:])
        # 动能通量 ∫u² dy
        integralU2 = np.sum(0.5 * (ua[integral_mask][1:]**2 + ua[integral_mask][:-1]**2) * dy_integral[1:])
        # 平方质量通量 ∫(u * alpha)² dy
        integral2 = np.sum(0.5 * (ua_alpha[1:]**2 + ua_alpha[:-1]**2) * dy_integral[1:])

        # 计算关键参数
        U = integralU2 / integralU if integralU != 0 else 0
        H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0


        results.append({
            "Time": time_v,
            "x": x,
            "U": U,
            "H": H,
            "ALPHA": ALPHA,
            "max_ya": max_ya
        })



    # 保存当前时间步的结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"integration_results_t{time_v}.csv"), index=False)
    print(f"Results saved for t={time_v} (head_x={head_x:.2f}m)")

    # --- 计算速度扰动 U_a' = U_a - U(x) ---
    # 从DataFrame中获取U值，并创建x到U的映射
    x_U_mapping = dict(zip(df['x'], df['U']))
    
    #U_perturb = np.zeros_like(Ua_A[0])
    U_perturb = Ua_A[0].copy()  # 直接使用 Ua_A[0] 的形状
    for x in x_coords:
        if x in x_U_mapping:  # 确保x在计算结果中
            mask = (X == x)
            U_perturb[mask] = Ua_A[0][mask] - x_U_mapping[x]

    # --- 插值到规则网格（可选，使绘图更平滑）---
    xi = np.linspace(X.min(), X.max(), 500)
    yi = np.linspace(Y.min(), Y.max(), 3000)
    xi, yi = np.meshgrid(xi, yi)
    U_perturb_i = griddata((X, Y), U_perturb, (xi, yi), method='linear')
    # 插值原始速度场（用于流线方向）
    uxi = griddata((X, Y), Ua_A[0], (xi, yi), method='linear')  # Ux 分量
    uyi = griddata((X, Y), Ua_A[1], (xi, yi), method='linear')  # Uy 分量
    alpha_i = griddata((X, Y), alpha_A, (xi, yi), method='linear')

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

    

    positions = {
        '1/6H': head_x - (1/6)*Height,
        '1/4H': head_x - 0.25*Height,
        '1/2H': head_x - 0.5*Height,
        'H': head_x - Height
    }
    y_text = 0.32
    
    

#def plot_velocity_vectors(X, Y, U_perturb, Ua_A, xi, yi, alpha_i, time_v, output_dir):
    
    """绘制速度矢量图"""
    plt.figure(figsize=FIG_SIZE)
    
    # 矢量场参数
    skip = 10 # 每隔10个点绘制一个箭头
    scale = 10
    width = 0.002
    
    plt.quiver(
        X[::skip], Y[::skip],
        U_perturb[::skip], Ua_A[1][::skip],
        scale=scale,
        width=width,
        color='blue'
    )
    
    if 'alpha_i' in locals():
        plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04    
    
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Velocity Vectors (t={time_v}) - No Interpolation')
    plt.savefig(os.path.join(output_dir, f'velocity_vectors_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()


    """绘制扰动速度流线图"""
    plt.figure(figsize=FIG_SIZE)
    
    # 归一化和缩放
    magnitude = np.sqrt(U_perturb_i**2 + uyi**2)
    U_perturb_i_norm = np.where(magnitude > 0, U_perturb_i / magnitude, 0)
    U_perturb_i_scaled = np.clip(U_perturb_i, -0.2, 0.05)
    
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


    """绘制原始速度流线图"""
    plt.figure(figsize=FIG_SIZE)
    uxi_scaled = np.clip(uxi, 0.1, 0.2)
    
    strm = plt.streamplot(
        xi, yi, uxi, uyi,
        color=uxi_scaled,
        cmap=plt.cm.rainbow,
        linewidth=1,
        density=3,
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
                       label="Velocity $U_a$ [m/s]")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Original Velocity Streamlines (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'origin_streamlines_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()


    """绘制Q准则涡分布图"""
    plt.figure(figsize=FIG_SIZE)
    
    Q_levels = np.linspace(-3.5, 3.5, 21)
    Q_contour = plt.contourf(
        xi, yi, Q_i,
        levels=Q_levels,
        cmap='bwr',
        extend='both'
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)
    

    
    
    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04
    
    cbar = plt.colorbar(Q_contour, label='Q-Criterion')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Q-Criterion Contours (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'Q_criterion_t{time_v}.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()


    """绘制优化后的Q准则涡分布图"""
    

    plt.figure(figsize=FIG_SIZE)
    
    Q_levels = np.linspace(-1.5, 1.5, 11)
    Q_contour_signed = plt.contourf(
        xi, yi, Q_signed_i,
        levels=Q_levels,
        cmap='bwr',
        extend='both'
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04
    
    cbar = plt.colorbar(Q_contour_signed, label='Q-Criterion')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)
    plt.title(f'Q-Criterion Contours_signed (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'Q_criterion_t{time_v}_signed.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()

    """绘制考虑涡量后的Q准则涡分布图"""
    

    plt.figure(figsize=FIG_SIZE)
    
    Q_levels = np.linspace(-10, 10, 41)
    Q_contour_signed = plt.contourf(
        xi, yi, Q_signed_smoothed_i,
        levels=Q_levels,
        cmap='bwr',
        extend='both'
    )
    
    plt.contour(xi, yi, alpha_i, **ALPHA_CONTOUR_PARAMS)

    for label, x_pos in positions.items():
        plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
        plt.text(x_pos + 0.01, y_text, f'{label}', 
                fontsize=15, zorder=3, color='b')
        #y_text += 0.04

    
    cbar = plt.colorbar(Q_contour_signed, label='Q-Criterion')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(*X_LIM)

        # 添加网格线设置
    ax = plt.gca()  # 获取当前坐标轴

    # 设置主要网格线（你指定的间隔）
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))  # 横向间隔0.1
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))  # 纵向间隔0.025

    # 设置次要网格线（可选，更密的网格）
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 横向次要间隔0.05
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.025))  # 纵向次要间隔0.0125

    # 启用网格线
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='gray', alpha=1.0)
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray', alpha=1.0)



    plt.title(f'Q-Criterion Contours_signed_omega (t={time_v})')
    plt.savefig(os.path.join(output_dir, f'Q_criterion_t{time_v}_signed_omega.png'), 
              dpi=300, bbox_inches='tight')
    plt.close()
    


print(f"All results saved to: {output_dir}")



#print(f"All results saved to: {output_dir}")
