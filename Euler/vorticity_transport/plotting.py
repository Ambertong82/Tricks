import os
import numpy as np


def plot_transport_terms(analyzer, data, time_v: float, head_x: float):
    """为每个 L 和 R 项生成 spanwise average (沿 z 平均) 的云图。"""
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

    # 1. 对网格和 alpha 场做 z 向平均, 得到 2D (x,y) 场
    X_avg = np.mean(data.X, axis=2)
    Y_avg = np.mean(data.Y, axis=2)
    alpha_avg = np.mean(data.alpha_A, axis=2)

    # 2. 准备所有待绘制的项 (包括总和项)
    L_total = np.zeros_like(data.vorticityUb)
    print("data.L_terms keys:", data.L_terms.keys())
    for k, v in data.L_terms.items():
        if k != "L0":
            L_total += v

    R_total = np.zeros_like(data.vorticityUb)
    print("data.R_terms keys:", data.R_terms.keys())
    for v in data.R_terms.values():
        R_total += v

    residual = L_total - R_total
    all_plots = {**data.L_terms, **data.R_terms, "L_sum": L_total, "R_sum": R_total, "Residual": residual}

    for name, field in all_plots.items():
        # 跳过验证项 L0 的绘图 (速度梯度矩阵)
        if name == "L0":
            continue

        # 若为向量 (3, nx, ny, nz)，取 z 分量后再做 spanwise average。
        if field.ndim == 4:
            plot_val = np.mean(field[2], axis=2)
            title_suffix = "(Z-component, spanwise-avg)"
        else:
            plot_val = np.mean(field, axis=2)
            title_suffix = "(Scalar, spanwise-avg)"

        plt.figure(figsize=analyzer.FIG_SIZE)

        # 用分位数限制极值，提升云图对比度。
        vlimit = np.percentile(np.abs(plot_val), 98)
        if vlimit == 0:
            vlimit = 1e-10

        # 增加 levels 让等值填充更平滑。
        levels = np.linspace(-vlimit, vlimit, 101)

        im = plt.contourf(
            X_avg,
            Y_avg,
            plot_val,
            levels=levels,
            cmap=analyzer.cmap,
            extend="both",
        )
        plt.contour(X_avg, Y_avg, alpha_avg, **analyzer.ALPHA_CONTOUR_PARAMS)
        plt.colorbar(im, label="Value", aspect=20)

        plt.xlim(*analyzer.X_LIM)
        plt.ylim(*analyzer.Y_LIM)
        plt.title(f"{name} {title_suffix} at t={time_v}")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        # 提高保存清晰度到 300 DPI
        save_name = os.path.join(plot_dir, f"contour_{name}.png")
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"All plots saved for t={time_v}")
