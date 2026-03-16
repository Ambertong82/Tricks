import os
import numpy as np


def plot_transport_terms(analyzer, data, time_v: float, head_x: float):
    """为每个 L 和 R 项生成 Z 方向切片的云图。"""
    import matplotlib.pyplot as plt

    # 统一设置字体大小
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )

    plot_dir = os.path.join(analyzer.output_dir, f"plots_t{time_v}")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Generating contour plots in {plot_dir}...")

    # 1. 找到最近的 Z 切面索引
    z_idx = np.argmin(np.abs(data.Z[0, 0, :] - analyzer.z_slice))
    print(f"Selected Z slice at z={data.Z[0, 0, z_idx]:.3f} (index {z_idx}) for plotting")
    real_z = data.Z[0, 0, z_idx]

    # 2. 提取切面网格
    X_slice = data.X[:, :, z_idx]
    Y_slice = data.Y[:, :, z_idx]
    alpha_slice = data.alpha_A[:, :, z_idx]

    # 3. 准备所有待绘制的项 (包括总和项)
    L_total = np.zeros_like(data.vorticityUb)
    for k, v in data.L_terms.items():
        if k != "L0":
            L_total += v

    R_total = np.zeros_like(data.vorticityUb)
    for v in data.R_terms.values():
        R_total += v

    all_plots = {**data.L_terms, **data.R_terms, "L_sum": L_total, "R_sum": R_total}

    for name, field in all_plots.items():
        # 跳过验证项 L0 的绘图 (速度梯度矩阵)
        if name == "L0":
            continue

        # 若为向量 (3, nx, ny, nz)，取 z 分量观察平面外旋转强度。
        if field.ndim == 4:
            plot_val = field[2, :, :, z_idx]
            title_suffix = "(Z-component)"
        else:
            plot_val = field[:, :, z_idx]
            title_suffix = "(Scalar)"

        plt.figure(figsize=analyzer.FIG_SIZE)

        # 用分位数限制极值，提升云图对比度。
        vlimit = np.percentile(np.abs(plot_val), 98)
        if vlimit == 0:
            vlimit = 1e-10

        # 增加 levels 让等值填充更平滑。
        levels = np.linspace(-vlimit, vlimit, 101)

        im = plt.contourf(
            X_slice,
            Y_slice,
            plot_val,
            levels=levels,
            cmap=analyzer.cmap,
            extend="both",
        )
        plt.contour(X_slice, Y_slice, alpha_slice, **analyzer.ALPHA_CONTOUR_PARAMS)
        plt.colorbar(im, label="Value", aspect=20)

        plt.xlim(*analyzer.X_LIM)
        plt.ylim(*analyzer.Y_LIM)
        plt.title(f"{name} {title_suffix} at t={time_v}, z={real_z:.3f}")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # 提高保存清晰度到 300 DPI
        save_name = os.path.join(plot_dir, f"contour_{name}.png")
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"All plots saved for t={time_v}")
