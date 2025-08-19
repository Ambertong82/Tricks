from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import fluidfoam

# 设置路径和参数
sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
X, Y, Z = fluidfoam.readmesh(sol)
times = [5]  # 选择的时间点
# 创建规则网格
xi = np.linspace(X.min(), X.max(), 100)
yi = np.linspace(Y.min(), Y.max(), 200)
xi, yi = np.meshgrid(xi, yi)


for i, time_v in enumerate(times):
    Ua_A = fluidfoam.readvector(sol, str(time_v), "U.a")
    Ux = Ua_A[0]
    Uy = Ua_A[1]
    
    alpha_A = fluidfoam.readscalar(sol, str(time_v), "alpha.a")



    # 插值到规则网格
    uxi = griddata((X, Y), Ux, (xi, yi), method='linear')
    uyi = griddata((X, Y), Uy, (xi, yi), method='linear')
    alpha_i = griddata((X, Y), alpha_A, (xi, yi), method='linear')

    # 归一化速度方向（仅用于颜色映射）
    magnitude = np.sqrt(uxi**2 + uyi**2)
    uxi_norm = np.where(magnitude > 0, uxi / magnitude, 0)
    color_scale = uxi_norm  # 颜色基于 Ux 方向 [-1, 1]

    # 创建画布
    plt.figure(figsize=(40, 3))  # 调整画布尺寸
    # --- 图层1: 绘制流线图（颜色基于方向） ---
    strm = plt.streamplot(
        xi, yi, uxi, uyi,
        color=color_scale,  # 颜色映射到 Ux 方向
        cmap=plt.cm.RdBu,   # 红蓝渐变色
        linewidth=0.5,      # 流线宽度
        density=5,          # 流线密度
        arrowsize=1,        # 箭头大小
        arrowstyle='->',    # 箭头样式
        zorder=2            # 确保在上层
    )

        # --- 图层2: 绘制 alpha.a = 0.001 的等值线 ---
    alpha_contour = plt.contour(
        xi, yi, alpha_i,
        levels=[0.001],     # 仅绘制 alpha=0.001 的等值线
        colors='black',     # 等值线颜色
        linewidths=2,       # 线宽
        linestyles='dashed', # 线型（可选）
        zorder=3            # 确保在最上层
    )


    # 添加流线颜色条
    cbar = plt.colorbar(
        strm.lines,
        label="Velocity Direction (Red=Positive X, Blue=Negative X)"
    )
    cbar.set_ticks([-1, 0, 1])

    # 标题和坐标轴
    plt.title(
        f"Streamlines + Alpha.a at Time = {time_v} s\n"
        f"Color: Velocity Direction (Red=Positive X, Blue=Negative X)",
        fontsize=12
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()
    plt.show()

#     # 绘制流线图
#     axes[i].streamplot(xi, yi, uxi, uyi, density=2, color='b', linewidth=1)
#     axes[i].set_title(f"Time = {time_v} s")
#     axes[i].set_xlabel("x (m)")
#     if i == 0:
#         axes[i].set_ylabel("y (m)")
#     axes[i].grid()

# plt.suptitle("Streamlines of Velocity Field at Different Times")
# plt.tight_layout()
# plt.show()
