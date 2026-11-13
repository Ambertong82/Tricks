"""二维 SedFoam：沿程浓度带速度、床面穿越通量、剪应力及水平固相通量。

浓度带速度为 int_band(U.a_x dy) / int_band(dy)，不按 alpha 加权。
每对相邻网格中心之间线性插值，故薄浓度带内可以没有网格中心。

定义 eta(x,t) 为从底部连续高浓度床层向上遇到的第一个 alpha=bed_alpha
交点。净固相体积通量 e（单位 m/s，正值指向床面上方）为

    e = bed_alpha * (Uy - Ux * d(eta)/dx - d(eta)/dt) |_(y=eta)

这是相对于移动的浓度等值线的净交换，不能标记泥沙的来源。
在床层向上首次遇到的 alpha.a=stress_alpha 交点插值 Taua、Taub；
沿当地切向/法向投影后计算 sqrt(tau_s_tan**2+tau_f_tan**2)。
Shields 指标的分母是 (rho_s-rho_f)*|g|*d。
另外逐 x 列计算 q_s=int_(-0.05)^top alpha.a*U.a_x dy (单位 m²/s)。
muI 带厚为从 muI_search_y_min 向上、muI=muI_low 与 muI=muI_high 两个等值线
第一个交点之间的高度差。
每个时刻只生成一份合并 CSV，包含完整 x 范围；behind_max 仅影响图像。
适用于 z 方向只有一层、网格坐标及各场使用相同单元顺序的算例。
"""

import os

import fluidfoam as fl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# ---- 1. 用户配置 ----
sol = "/media/amber/Elements/Scour/vanTC/equalibrium/case1031_6"
output_dir = "/home/amber/postpro/Scour/vanTC/shields/alpha_band_shear_shields316"
times = np.arange(17, 40, 2)  # 实际为 33, 35, 37, 39, 41 s

alpha_low = 0.08
alpha_high = 0.57
bed_alpha = 0.57
head_alpha = 0.001
head_y_min = -0.015
head_dt = 2.0                # eta_t 优先中心差分，头速优先前向差分
behind_max = 2.5            # 只画头部后方 1.5 m；CSV 输出所有 x
round_x = 9
stress_alpha = 0.08         # 沿程剪应力提取的固相体积分数
muI_field = "muI"           # 惯性数场 (MuI 摩擦模型输出)，无量纲
muI_low = 0.41              # 惯性数带的下界
muI_high = 0.85             # 惯性数带的上界
muI_search_y_min = -0.05    # 从该 y 起向上找两个等值线的第一个交点
solid_stress_field = "Taua" # phasea: 泥沙应力张量，Pa
fluid_stress_field = "Taub" # phaseb: 流体应力张量，Pa
d_m = 0.1e-3              # 颗粒直径 (m)；按当前算例修改，例如 0.1 mm 写 0.1e-3
rho_s = 1530.0             # 泥沙密度 (kg/m3)，核对 constant/transportProperties
rho_f = 1000.0             # 水密度 (kg/m3)
g_magnitude = 9.81         # 重力矢量的模 (m/s2)，不是坡向 gy
flux_y_min = -0.05         # 水平固相通量积分下限 (m)
flux_y_top = 0.25          # 当前算例顶部 y 坐标 (m)；换网格时同步修改


def tname(t):
    return f"{float(t):g}"


def mean_velocity_in_band(y, a, u, low, high):
    """逐段截取线性浓度带；返回(Ux 垂向均值, 带厚, 区段数)。"""
    integral = 0.0
    thickness = 0.0
    nsegments = 0
    for i in range(len(y) - 1):
        dy = y[i + 1] - y[i]
        a0, a1 = a[i], a[i + 1]
        u0, u1 = u[i], u[i + 1]
        if dy <= 0 or not np.all(np.isfinite([a0, a1, u0, u1])):
            continue
        da = a1 - a0
        if abs(da) < 1e-14:
            if not (low <= a0 <= high):
                continue
            left, right = 0.0, 1.0
        else:
            f_low = (low - a0) / da
            f_high = (high - a0) / da
            left = max(0.0, min(f_low, f_high))
            right = min(1.0, max(f_low, f_high))
            if right <= left:
                continue
        h = dy * (right - left)
        u_left = u0 + (u1 - u0) * left
        u_right = u0 + (u1 - u0) * right
        integral += h * (u_left + u_right) / 2.0
        thickness += h
        nsegments += 1
    if thickness == 0.0:
        return np.nan, 0.0, 0
    return integral / thickness, thickness, nsegments


def crossing_thickness(y, values, low, high, y_min):
    """从 y_min 起向上找 values=low 与 values=high 的第一个交点，返回高度差 (m)。

    网格中心之间线性插值，交点落在两个网格中心之间也能算准；
    先碰到哪个交点就停哪个。任一交点找不到时返回 nan。
    """
    if len(y) < 2:
        return np.nan
    found = {}
    for i in range(len(y) - 1):
        if y[i + 1] <= y_min:
            continue
        f0, f1 = values[i], values[i + 1]
        if y[i + 1] <= y[i] or not np.all(np.isfinite([f0, f1])) or f1 == f0:
            continue
        for level in (low, high):
            if level in found:
                continue
            if (f0 - level) * (f1 - level) <= 0.0:
                f = (level - f0) / (f1 - f0)
                found[level] = y[i] + f * (y[i + 1] - y[i])
        if low in found and high in found:
            break
    if low not in found or high not in found:
        return np.nan
    return abs(found[high] - found[low])


def bed_crossing(y, a, level, ux=None, uy=None):
    """从底部床层向上找第一处高于 level 到低于 level 的交点。"""
    no_crossing = (np.nan, np.nan, np.nan)
    if len(y) < 2 or not np.isfinite(a[0]) or a[0] < level:
        return no_crossing
    for i in range(len(y) - 1):
        a0, a1 = a[i], a[i + 1]
        if not np.all(np.isfinite([y[i], y[i + 1], a0, a1])):
            return no_crossing
        if a0 >= level and a1 < level and y[i + 1] > y[i]:
            f = (level - a0) / (a1 - a0)
            eta = y[i] + f * (y[i + 1] - y[i])
            uxi = np.nan if ux is None else ux[i] + f * (ux[i + 1] - ux[i])
            uyi = np.nan if uy is None else uy[i] + f * (uy[i + 1] - uy[i])
            return float(eta), float(uxi), float(uyi)
        if a1 < level:  # 床层必须与底部连续，不能跨越稀疏区追逐上方浓团
            return no_crossing
    return no_crossing


def x_derivative_on_valid_runs(x, values):
    """在连续有效网格列上求 eta_x；不跨过缺失床面的区段插值。"""
    derivative = np.full(len(x), np.nan)
    valid = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, stop in zip(edges[::2], edges[1::2]):
        count = stop - start
        if count >= 2:
            derivative[start:stop] = np.gradient(
                values[start:stop], x[start:stop], edge_order=2 if count >= 3 else 1
            )
    return derivative


def integrate_signed_crossing(x, e, valid):
    """积分相邻有效点的 e 正/负部分；不跨越 NaN 或零交点。"""
    positive = 0.0
    negative = 0.0
    for i in range(len(x) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        e0, e1 = e[i], e[i + 1]
        dx = x[i + 1] - x[i]
        if dx <= 0:
            continue
        if e0 * e1 < 0:
            f = abs(e0) / (abs(e0) + abs(e1))
            if e0 > 0:
                positive += e0 * dx * f / 2
                negative += -e1 * dx * (1 - f) / 2
            else:
                negative += -e0 * dx * f / 2
                positive += e1 * dx * (1 - f) / 2
        else:
            average = (e0 + e1) * dx / 2
            if average >= 0:
                positive += average
            else:
                negative -= average
    return positive, negative


def integrate_solid_flux_column(y, a, ux, y_min, y_top=None):
    """积分 alpha*U.a_x dy；网格中心之间两场线性插值，积为二次函数。"""
    if (len(y) < 2 or not np.all(np.diff(y) > 0)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(ux))):
        return np.nan
    bottom_face = y[0] - 0.5 * (y[1] - y[0])
    top_face = y[-1] + 0.5 * (y[-1] - y[-2])
    low = max(float(y_min), bottom_face)
    high = top_face if y_top is None else float(y_top)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.nan

    total = max(0.0, min(high, y[0]) - low) * a[0] * ux[0]
    for i in range(len(y) - 1):
        left, right = max(low, y[i]), min(high, y[i + 1])
        if right <= left:
            continue
        f0 = (left - y[i]) / (y[i + 1] - y[i])
        f1 = (right - y[i]) / (y[i + 1] - y[i])
        fm = (f0 + f1) / 2
        da, du = a[i + 1] - a[i], ux[i + 1] - ux[i]
        q0 = (a[i] + f0 * da) * (ux[i] + f0 * du)
        qm = (a[i] + fm * da) * (ux[i] + fm * du)
        q1 = (a[i] + f1 * da) * (ux[i] + f1 * du)
        total += (right - left) * (q0 + 4 * qm + q1) / 6
    if high > y[-1]:
        total += max(0.0, high - max(low, y[-1])) * a[-1] * ux[-1]
    return total


def main():
    denominator = (rho_s - rho_f) * g_magnitude * d_m
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("要求 rho_s > rho_f、g_magnitude > 0、d_m > 0")
    # ---- 2. 网格只读一次；先按 x 分列，再按 y 从低到高排列 ----
    X, Y, Z = fl.readmesh(sol, verbose=False)
    X, Y, Z = (np.asarray(v).ravel() for v in (X, Y, Z))
    n_cells = X.size
    if Y.size != n_cells or Z.size != n_cells:
        raise ValueError("网格坐标长度不一致")
    if np.unique(np.round(Z, round_x)).size != 1:
        raise ValueError("本脚本只适用于 z 方向一层的二维算例")

    x_columns, group = np.unique(np.round(X, round_x), return_inverse=True)
    order = np.lexsort((Y, group))
    counts = np.bincount(group, minlength=x_columns.size)
    starts = np.r_[0, np.cumsum(counts)]
    column_ids = [order[starts[j]:starts[j + 1]] for j in range(x_columns.size)]

    alpha_cache = {}
    eta_cache = {}
    muI_cache = {}

    def field_exists(t, name):
        base = os.path.join(sol, tname(t), name)
        return os.path.isfile(base) or os.path.isfile(base + ".gz")

    def read_alpha(t):
        key = tname(t)
        if key not in alpha_cache:
            a = np.asarray(fl.readscalar(sol, key, "alpha.a", verbose=False)).ravel()
            if a.size != n_cells:
                raise ValueError(f"{key}/alpha.a 单元数与网格不一致")
            alpha_cache[key] = a
        return alpha_cache[key]

    def read_muI(t):
        key = tname(t)
        if key not in muI_cache:
            v = np.asarray(fl.readscalar(sol, key, muI_field, verbose=False)).ravel()
            if v.size != n_cells:
                raise ValueError(f"{key}/{muI_field} 单元数与网格不一致")
            muI_cache[key] = v
        return muI_cache[key]

    def read_velocity(t):
        U = np.asarray(fl.readvector(sol, tname(t), "U.a", verbose=False))
        if U.shape == (3, n_cells):
            return U[0], U[1]
        if U.shape == (n_cells, 3):
            return U[:, 0], U[:, 1]
        raise ValueError(f"{tname(t)}/U.a 形状异常：{U.shape}")

    def read_tau_tensor(t, field_name):
        """volTensorField 的 9 分量顺序为 xx,xy,xz,yx,yy,yz,zx,zy,zz。"""
        tau = np.asarray(fl.readtensor(sol, tname(t), field_name, verbose=False))
        if tau.shape == (9, n_cells):
            return tau
        if tau.shape == (n_cells, 9):
            return tau.T
        raise ValueError(f"{tname(t)}/{field_name} 张量形状异常：{tau.shape}")

    def bed_surface(t):
        key = tname(t)
        if key not in eta_cache:
            a = read_alpha(t)
            eta = np.full(x_columns.size, np.nan)
            for j, ids in enumerate(column_ids):
                eta[j] = bed_crossing(Y[ids], a[ids], bed_alpha)[0]
            eta_cache[key] = eta
        return eta_cache[key]

    def find_head(a):
        mask = (Y > head_y_min) & (a > head_alpha)
        if not np.any(mask):
            raise ValueError("找不到头部，检查 head_alpha 和 head_y_min")
        return float(np.max(X[mask]))

    os.makedirs(output_dir, exist_ok=True)
    fig, (ax_abs, ax_rel, ax_bed, ax_e) = plt.subplots(
        4, 1, figsize=(10, 14)
    )
    head_positions = []
    # 带厚热图用的原始数据：每个时间一行，距离随头部位置变化
    heat_times = []
    heat_distances = []
    heat_thickness = []
    heat_muI_thickness = []
    fig_tau, (ax_tau, ax_shields) = plt.subplots(2, 1, figsize=(10, 8))

    # ---- 3. 各时刻求速度带、床面、eta_t、eta_x 和 e ----
    for t in times:
        required_fields = ("alpha.a", "U.a", solid_stress_field,
                           fluid_stress_field, muI_field)
        missing = [name for name in required_fields if not field_exists(t, name)]
        if missing:
            print(f"t={tname(t)}: 缺少 {', '.join(missing)}，跳过")
            continue
        a = read_alpha(t)
        mu_i = read_muI(t)
        ux, uy = read_velocity(t)
        tau_s = read_tau_tensor(t, solid_stress_field)
        tau_f = read_tau_tensor(t, fluid_stress_field)
        eta = bed_surface(t)
        x_head = find_head(a)

        mean_ux = np.full(x_columns.size, np.nan)
        band_thickness = np.zeros(x_columns.size)
        muI_thickness = np.full(x_columns.size, np.nan)
        ux_bed = np.full(x_columns.size, np.nan)
        uy_bed = np.full(x_columns.size, np.nan)
        y_stress = np.full(x_columns.size, np.nan)
        q_s = np.full(x_columns.size, np.nan)
        # 在 alpha=0.08 交点处读取 xx,xy,yx,yy 全部平面应力分量。
        tau_s_plane = np.full((4, x_columns.size), np.nan)
        tau_f_plane = np.full((4, x_columns.size), np.nan)
        for j, ids in enumerate(column_ids):
            y_col, a_col = Y[ids], a[ids]
            mean_ux[j], band_thickness[j], _ = mean_velocity_in_band(
                y_col, a_col, ux[ids], alpha_low, alpha_high
            )
            muI_thickness[j] = crossing_thickness(
                y_col, mu_i[ids], muI_low, muI_high, muI_search_y_min
            )
            q_s[j] = integrate_solid_flux_column(
                y_col, a_col, ux[ids], flux_y_min, flux_y_top
            )
            _, ux_bed[j], uy_bed[j] = bed_crossing(
                y_col, a_col, bed_alpha, ux[ids], uy[ids]
            )
            y_stress[j] = bed_crossing(y_col, a_col, stress_alpha)[0]
            if np.isfinite(y_stress[j]):
                for k, component in enumerate((0, 1, 3, 4)):
                    tau_s_plane[k, j] = np.interp(
                        y_stress[j], y_col, tau_s[component, ids]
                    )
                    tau_f_plane[k, j] = np.interp(
                        y_stress[j], y_col, tau_f[component, ids]
                    )

        slope_08 = x_derivative_on_valid_runs(x_columns, y_stress)
        m = slope_08
        # n=(-m,1)/sqrt(1+m²), t=(1,m)/sqrt(1+m²)
        # t·(Tau·n) = (-m Tau_xx + Tau_xy -m² Tau_yx +m Tau_yy)/(1+m²)
        def projected_tangent(plane):
            xx, xy, yx, yy = plane
            return (-m * xx + xy - m * m * yx + m * yy) / (1 + m * m)

        tau_s_tan = projected_tangent(tau_s_plane)
        tau_f_tan = projected_tangent(tau_f_plane)
        tau_tangent_magnitude = np.hypot(tau_s_tan, tau_f_tan)
        shields_tangent = tau_tangent_magnitude / denominator
        ax_tau.plot(x_columns, tau_tangent_magnitude, lw=1.4,
                    label=f"t={tname(t)} s")
        ax_shields.plot(x_columns, shields_tangent, lw=1.4,
                        label=f"t={tname(t)} s")

        previous_t, next_t = float(t) - head_dt, float(t) + head_dt
        has_previous = field_exists(previous_t, "alpha.a")
        has_next = field_exists(next_t, "alpha.a")
        eta_t = np.full(x_columns.size, np.nan)
        if has_previous and has_next:
            eta_t = (bed_surface(next_t) - bed_surface(previous_t)) / (2 * head_dt)
            eta_scheme = "centered"
        elif has_next:
            eta_t = (bed_surface(next_t) - eta) / head_dt
            eta_scheme = "forward"
        elif has_previous:
            eta_t = (eta - bed_surface(previous_t)) / head_dt
            eta_scheme = "backward"
        else:
            eta_scheme = "unavailable"

        if has_next:
            u_head = (find_head(read_alpha(next_t)) - x_head) / head_dt
        elif has_previous:
            u_head = (x_head - find_head(read_alpha(previous_t))) / head_dt
        else:
            u_head = np.nan

        eta_x = x_derivative_on_valid_runs(x_columns, eta)
        e = bed_alpha * (uy_bed - ux_bed * eta_x - eta_t)
        # 正号：向上离开 alpha=0.57 床面；负号：进入床面以下。

        distance = x_head - x_columns
        relative_ux = mean_ux - u_head
        csv_table = np.column_stack((
            x_columns, distance, mean_ux, relative_ux, band_thickness,
            muI_thickness,
            np.full(x_columns.size, x_head),
            np.full(x_columns.size, u_head), eta,
            eta_x, eta_t, e, tau_tangent_magnitude, shields_tangent, q_s,
        ))
        csv_name = os.path.join(output_dir, f"band_speed_bed_exchange_t{tname(t)}.csv")
        np.savetxt(
            csv_name, csv_table, delimiter=",", fmt="%.10g", comments="",
            header=(
                "x_m,distance_behind_head_m,mean_Uax_m_s,mean_Uax_minus_head_m_s,"
                "band_thickness_m,muI_thickness_m,head_x_m,head_speed_m_s,"
                "eta_alpha057_m,deta_dx,deta_dt_m_s,e_positive_up_m_s,"
                "tau_tangent_hypot_Pa,shields_tangent,q_s_m2_s"
            ),
        )

        show = (distance >= 0) & (distance <= behind_max)
        show_band = show & np.isfinite(mean_ux)
        show_bed = show & np.isfinite(eta)
        show_e = show & np.isfinite(e)
        ax_abs.plot(x_columns[show_band], mean_ux[show_band],
                    lw=1.4, label=f"t={tname(t)} s")
        ax_rel.plot(distance[show_band], relative_ux[show_band],
                    lw=1.4, label=f"t={tname(t)} s; Uhead={u_head:.3f} m/s")
        heat_times.append(float(t))
        heat_distances.append(distance.copy())
        heat_thickness.append(band_thickness.copy())
        heat_muI_thickness.append(muI_thickness.copy())
        ax_bed.plot(x_columns, np.where(show_bed, eta, np.nan),
                    lw=1.4, label=f"t={tname(t)} s")
        ax_e.plot(distance, np.where(show_e, e, np.nan),
                  lw=1.4, label=f"t={tname(t)} s")

        E_up, D_down = integrate_signed_crossing(x_columns, e, show_e)
        head_positions.append(x_head)
        print(f"t={tname(t)}: x_head={x_head:.4f} m, eta_t={eta_scheme}, "
              f"有效床面列={np.isfinite(eta).sum()}, e有效列={show_e.sum()}, "
              f"E_up={E_up:.5g}, D_down={D_down:.5g} m2/s; "
              f"alpha={stress_alpha:g} 投影有效列={np.isfinite(shields_tangent).sum()} "
              f"-> {csv_name}")

    if not head_positions:
        raise RuntimeError("times 中没有同时包含 alpha.a、U.a、Taua 和 Taub 的有效场")

    ax_tau.set(xlabel="Streamwise x (m)",
               ylabel=r"$\sqrt{\tau_{s,\parallel}^{2}+\tau_{f,\parallel}^{2}}$ (Pa)")
    ax_tau.set_title(f"Local tangent at alpha.a={stress_alpha:g}; d={d_m * 1e3:g} mm")
    ax_shields.set(xlabel="Streamwise x (m)",
                   ylabel=r"$\Theta_{\parallel}=\tau_{\parallel}/[(\rho_s-\rho_f)g d]$")
    for ax in (ax_tau, ax_shields):
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig_tau.tight_layout()
    stress_png = os.path.join(output_dir, "shear_shields_along_x.png")
    fig_tau.savefig(stress_png, dpi=200)
    plt.close(fig_tau)

    # ---- 4. 原有三张图，加床面高度和相对床面的穿越通量 ----
    ax_abs.set(xlabel="Streamwise x (m)",
               ylabel=r"Vertical mean $\overline{U}_{a,x}$ (m/s)")
    ax_abs.set_xlim(min(head_positions) - behind_max - 0.01,
                    max(head_positions) + 0.01)
    ax_rel.axhline(0, color="black", lw=1)
    ax_rel.set(xlabel="Distance behind head (m)",
               ylabel=r"$\overline{U}_{a,x}-U_{head}$ (m/s)", xlim=(0, behind_max))
    ax_bed.set(xlabel="Streamwise x (m)",
               ylabel=r"Bed $\eta$ at $\alpha_a=0.57$ (m)")
    ax_bed.set_xlim(min(head_positions) - behind_max - 0.01,
                    max(head_positions) + 0.01)
    ax_e.axhline(0, color="black", lw=1)
    ax_e.set(xlabel="Distance behind head (m)",
             ylabel=r"Net exchange $e$ (+ upward) (m/s)", xlim=(0, behind_max))
    for ax in (ax_abs, ax_rel, ax_bed, ax_e):
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{alpha_low:g} <= alpha.a <= {alpha_high:g}; "
                 f"bed contour alpha.a={bed_alpha:g}")
    fig.tight_layout()
    png_name = os.path.join(output_dir, "band_speed_bed_exchange_evolution.png")
    fig.savefig(png_name, dpi=200)
    plt.close(fig)
    print(f"床面图像: {png_name}")
    print(f"沿程双相应力与 Shields 指标图像: {stress_png}")

    # ---- 5. 带厚热图：横轴头部后方距离，纵轴时间，颜色为带厚 ----
    # 上：alpha 浓度带厚；下：muI 带厚（0.41 与 0.88 两个交点的高度差）。
    # 每个时刻的头部位置不同，distance 网格跟着变，所以先插值到统一的距离网格。
    if heat_times:
        dist_grid = np.linspace(0.0, behind_max, 201)
        time_axis = np.asarray(heat_times)

        def thickness_grid(thickness_list):
            grid = np.full((len(heat_times), dist_grid.size), np.nan)
            for row, (dist, thick) in enumerate(
                    zip(heat_distances, thickness_list)):
                valid = (dist >= 0.0) & (thick > 0.0)
                if not np.any(valid):
                    continue
                d = dist[valid]
                th = thick[valid]
                order = np.argsort(d)
                # 左端用第一个有效值补齐：头部列的 distance 是 x_head-x 的浮点残差
                # (~1e-16)，会小于网格点 0.0，若用 left=nan 就会留下空白条。
                # 右端保持 NaN：带结束之后本来就没有数据。
                grid[row] = np.interp(dist_grid, d[order], th[order],
                                      left=th[order][0], right=np.nan)
            return grid

        thick_grid = thickness_grid(heat_thickness)
        muI_grid = thickness_grid(heat_muI_thickness)

        # 单色相顺序色阶：浅 = 薄，深 = 厚
        blue_ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                     "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                     "#184f95", "#104281", "#0d366b"]
        cmap = LinearSegmentedColormap.from_list("band_thickness", blue_ramp)

        fig_heat, (ax_band, ax_muI) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, sharey=True
        )
        for ax, grid, title in (
            (ax_band, thick_grid,
             f"Alpha band thickness ({alpha_low:g} <= alpha.a <= {alpha_high:g})"),
            (ax_muI, muI_grid,
             f"muI band thickness ({muI_low:g} <= muI <= {muI_high:g})"),
        ):
            mesh = ax.pcolormesh(dist_grid, time_axis, grid,
                                 cmap=cmap, shading="auto")
            ax.set_facecolor("#f0efec")  # 无带的单元显示为浅灰
            cbar = fig_heat.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Thickness (m)")
            ax.set_title(title, fontsize=10)
        ax_muI.set(xlabel="Distance behind head (m)", ylabel="Time (s)")
        fig_heat.suptitle("Alpha band vs muI band thickness", fontsize=12)
        fig_heat.tight_layout()
        heat_png = os.path.join(output_dir, "band_vs_muI_thickness_heatmap.png")
        fig_heat.savefig(heat_png, dpi=200)
        plt.close(fig_heat)
        print(f"带厚热图: {heat_png}")


if __name__ == "__main__":
    main()