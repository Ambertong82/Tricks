import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 设置路径和输出目录
#sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
#sol = "/home/amber/OpenFOAM/amber-v2306/Marino/case0704_4"
output_dir = "/home/amber/postpro/u_umean_saline"
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
#times  = [20,21,22]
times = np.arange(5, 22, 1)

#for time_v in times:
for i in range(len(times)):
    time_v = times[i]
    # 读取场数据
    Ua_A = fluidfoam.readvector(sol, str(time_v), "U.a")
    alpha_A = fluidfoam.readscalar(sol, str(time_v), "alpha.a")

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

   # 归一化速度方向（用于颜色映射）
    magnitude = np.sqrt(uxi**2 + uyi**2)
    uxi_norm = np.where(magnitude > 0, uxi / magnitude, 0)
    #uxi_norm = np.where(uxi> 0, 1)  # 归一化到 [-1, 1]
    color_scale = uxi_norm  # 颜色基于 Ux 方向 [-1, 1]
    # 手动缩放数据
    vmin, vmax = -0.2, 0.1  # 根据数据范围调整
    U_perturb_i_scaled = np.clip(U_perturb_i, vmin, vmax)
    plt.rcParams.update({
    'font.size': 28,          # 全局字体大小
    'axes.titlesize': 28,     # 子图标题大小
    'axes.labelsize': 24,     # 坐标轴标签大小
    'xtick.labelsize': 24,    # x轴刻度标签大小
    'ytick.labelsize': 24,    # y轴刻度标签大小
    'legend.fontsize': 24     # 图例字体大小
})

# 创建画布
    plt.figure(figsize=(40, 6))  # 长条形画布适合流动显示

# --- 图层1: 绘制扰动速度场（pcolormesh）---
    strm=plt.streamplot(
    xi, yi, U_perturb_i,uyi,
    #shading='auto',
    color=U_perturb_i_scaled,  # 使用扰动速度作为颜色
    cmap=plt.cm.rainbow,  # 红蓝渐变色（反向：正值为红，负值为蓝）
    linewidth=1,      # 流线宽度
    density=5,          # 流线密度
    arrowsize=2,        # 箭头大小
    arrowstyle='->',    # 箭头样式
    #alpha=0.7,  # 半透明
    zorder=1,
    #vmin=-0.2, vmax=0.2, # 颜色映射范围（根据数据调整）
    )
    
    alpha_contour = plt.contour(
        xi, yi, alpha_i,
        levels=[1e-4],     # 仅绘制 alpha=0.001 的等值线
        colors='black',     # 等值线颜色
        linewidths=2,       # 线宽
        linestyles='dashed', # 线型（可选）
        zorder=3            # 确保在最上层
    )
    #cbar.set_ticks([-0.2, 0, 0.5])
    cbar = plt.colorbar(
    strm.lines,
    label="Velocity Perturbation $U_a\'$ [m/s]"
    )
    #cbar.set_ticks([-0.5, 0, 0.05])
    
    
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(1, 2.8)
    plt.title(f'Perturbation Velocity Streamlines (t={time_v})')
    #plt.show()

#     # # --- 图层3: 添加头部位置标记 ---
#     # head_x = df['x'].max()  # 从之前的结果获取头部位置
#     # plt.axvline(x=head_x, color='k', linestyle='--', linewidth=1.5, zorder=3)
#     # plt.text(head_x+0.1, yi.min()+0.1, 'Head Position', fontsize=10, zorder=3)

# 保存图像
    plt.savefig(os.path.join(output_dir, f'perturbation_streamlines_t{time_v}.png'), 
            dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


# 创建画布
    plt.figure(figsize=(40, 6))

# --- 绘制原始速度场的流线图 ---
    strm = plt.streamplot(
    xi, yi, uxi, uyi,
    color=uxi,  # 颜色映射到 Ux 方向
    cmap=plt.cm.rainbow,   # 红蓝渐变色（左蓝右红）
    linewidth=1,
    density=5,
    arrowsize=2,
    arrowstyle='->',
    zorder=2            # 确保在上层
    )

    alpha_contour = plt.contour(
        xi, yi, alpha_i,
        levels=[1e-4],     # 仅绘制 alpha=0.001 的等值线
        colors='black',     # 等值线颜色
        linewidths=2,       # 线宽
        linestyles='dashed', # 线型（可选）
        zorder=3            # 确保在最上层
    )
    
    cbar = plt.colorbar(
    strm.lines,
    label="Velocity Direction (Red=Positive X, Blue=Negative X)"
    )
    #cbar.set_ticks([-1, 0, 1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(X.min(), 2.8)
    plt.title(f'Original Velocity Streamlines (t={time_v})')


# # --- 图层3: 添加头部位置标记 ---
#     head_x = df['x'].max()  # 从之前的结果获取头部位置
#     plt.axvline(x=head_x, color='k', linestyle='--', linewidth=1.5, zorder=3)
#     plt.text(head_x+0.1, yi.min()+0.1, 'Head Position', fontsize=10, zorder=3)

    

# 保存图像
    plt.savefig(os.path.join(output_dir, f'origin_streamlines_t{time_v}.png'), 
            dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


    # 创建画布和子图（1行2列）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(80, 24))


# --- 子图1: 扰动速度场流线 ---
    strm1 = ax1.streamplot(
    xi, yi, U_perturb_i, uyi,
    color=U_perturb_i,
    cmap=plt.cm.RdBu,
    linewidth=3,
    density=5,
    arrowsize=3.5,
    zorder = 2,

    )
    alpha_contour = ax1.contour(
        xi, yi, alpha_i,
        levels=[1e-4],     # 仅绘制 alpha=0.001 的等值线
        colors='black',     # 等值线颜色
        linewidths=3,       # 线宽
        linestyles='dashed', # 线型（可选）
        zorder=3            # 确保在最上层
    )
    fig.colorbar(strm1.lines, ax=ax1, label='Perturbation Velocity $U_a\'$ [m/s]')
    ax1.set_title('Perturbation Velocity Streamlines')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_xlim(X.min(), 2.8)
    #ax1.axvline(x=head_x, color='k', linestyle='--', linewidth=1.5, zorder=3)

# --- 子图2: 原始速度场流线 ---
    strm2 = ax2.streamplot(
    xi, yi, uxi, uyi,
    color=uxi_norm,
    cmap=plt.cm.rainbow,
    linewidth=3,
    density=5,
    arrowsize=3.5,
    zorder=2
    )
    alpha_contour = ax2.contour(
        xi, yi, alpha_i,
        levels=[1e-4],     # 仅绘制 alpha=0.001 的等值线
        colors='black',     # 等值线颜色
        linewidths=3,       # 线宽
        linestyles='dashed',# 线型（可选）
        zorder=3,         # 确保在最上层

    )
    cbar = fig.colorbar(strm2.lines, ax=ax2, label='Normalized Ux Direction')
    ax2.set_title('Original Velocity Streamlines')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_xlim(X.min(), 2.8)
    #cbar.set_climits([-0.5,  0.5])
    #ax2.axvline(x=head_x, color='k', linestyle='--', linewidth=1.5, zorder=3)

# 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'combined_streamlines_t{time_v}.png'), 
            dpi=100, bbox_inches='tight')
    #plt.show()
    plt.close()

    
    


print(f"All results saved to: {output_dir}")



#print(f"All results saved to: {output_dir}")
