import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import binary_closing, gaussian_filter, label

plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})


@dataclass
class PlotStyle:
    fig_size: Tuple[int, int] = (40, 6)
    x_lim: Tuple[float, float] = (0.0, 1.6)
    y_lim: Tuple[float, float] = (0.0, 0.3)
    alpha_contour: Dict = field(default_factory=lambda: {
        "levels": [1e-5],
        "colors": "black",
        "linewidths": 2,
        "linestyles": "dashed",
        "zorder": 3,
    })
    marker_color: str = "fuchsia"


@dataclass
class AnalyzerConfig:
    sol: str = '/media/amber/53EA-E81F/PhD/case231020_5'
    # sol: str = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/3d/case091020_5"
    # output_dir: str = "/home/amber/postpro/POD/u_umean_tc2dcoarse"
    output_dir: str = "/home/amber/postpro/POD/u_umean_coarse_tc3dmiddle"
    alpha_threshold: float = 1e-5
    y_min: float = 0.0
    times: Iterable[int] = (5,7, 10)
    head_height: float = 0.3
    q_threshold: float = 0.5
    lambda2_threshold: float = -0.1
    interp_shape: Tuple[int, int] = (750, 200)
    closing_kernel: Tuple[int, int] = (10, 10)


class ManualPCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, data: np.ndarray) -> "ManualPCA":
        data = np.asarray(data)
        self.mean_ = data.mean(axis=0)
        cov = np.cov(data - self.mean_, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        self.components_ = eigvecs[:, order][:, : self.n_components]
        self.explained_variance_ = eigvals[order][: self.n_components]
        return self

    def properties(self) -> Dict[str, np.ndarray | float]:
        major_axis = self.components_[:, 0]
        if major_axis[0] < 0:
            major_axis = -major_axis
        angle = np.arctan2(major_axis[1], major_axis[0])
        return {
            "center": self.mean_,
            "major_axis": major_axis,
            "minor_axis": self.components_[:, 1],
            "length": 4 * np.sqrt(self.explained_variance_[0]),
            "width": 4 * np.sqrt(self.explained_variance_[1]),
            "angle": angle,
        }


class TurbidityCurrentAnalyzer:
    def __init__(self, config: AnalyzerConfig | None = None, style: PlotStyle | None = None):
        self.cfg = config or AnalyzerConfig()
        self.style = style or PlotStyle()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_q = os.path.join(self.cfg.output_dir, f"vortex_properties_{self.timestamp}.csv")
        self.csv_l2 = os.path.join(self.cfg.output_dir, f"lambda2_vortex_properties_{self.timestamp}.csv")
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    # ---------- 数值工具 ---------- #

    @staticmethod
    def q_criterion(dUx, dUy, dVx, dVy):
        s_xx = dUx
        s_xy = 0.5 * (dUy + dVx)
        s_yy = dVy
        omega_xy = 0.5 * (dVx - dUy)
        s_norm = s_xx**2 + 2 * s_xy**2 + s_yy**2
        omega_norm = 2 * omega_xy**2
        return np.nan_to_num(0.5 * (omega_norm - s_norm), nan=0.0)

    @staticmethod
    def lambda2_field(ux, uy, dx, dy):
        dudx = np.gradient(ux, dx, axis=1)
        dudy = np.gradient(ux, dy, axis=0)
        dvdx = np.gradient(uy, dx, axis=1)
        dvdy = np.gradient(uy, dy, axis=0)
        lambda2 = np.empty_like(ux)
        for i in range(ux.shape[0]):
            for j in range(ux.shape[1]):
                s = np.array([[dudx[i, j], 0.5 * (dudy[i, j] + dvdx[i, j])],
                              [0.5 * (dudy[i, j] + dvdx[i, j]), dvdy[i, j]]])
                omega = np.array([[0.0, 0.5 * (dudy[i, j] - dvdx[i, j])],
                                  [-0.5 * (dudy[i, j] - dvdx[i, j]), 0.0]])
                m = s @ s + omega @ omega
                lambda2[i, j] = np.sort(np.linalg.eigvalsh(m))[1]
        return lambda2

    @staticmethod
    def integrate_column(y, u, alpha, threshold=1e-3):
        sign_idx = np.where(np.diff(np.sign(u)))[0]
        cutoff = len(y) - 1
        for idx in sign_idx:
            if y[idx] > threshold and u[idx] > 0 and u[idx + 1] < 0:
                cutoff = idx + 1
                break
        u_alpha = u * alpha
        int_u = np.trapz(u[:cutoff], y[:cutoff])
        int_u2 = np.trapz(u[:cutoff] ** 2, y[:cutoff])
        int_ua = np.trapz(u_alpha[:cutoff], y[:cutoff])
        int_ua2 = np.trapz(u_alpha[:cutoff] ** 2, y[:cutoff])
        safe = lambda a, b: a / b if b != 0 else 0.0
        return {
            "U": safe(int_u2, int_u),
            "H": safe(int_u**2, int_u2),
            "ALPHA": safe(int_ua, int_u),
            "H_alpha": safe(int_ua**2, int_ua2),
            "zero_y": y[cutoff],
        }

    # ---------- 主流程 ---------- #

    def run_analysis(self):
        X, Y, Z = fluidfoam.readmesh(self.cfg.sol)
        for time_v in self.cfg.times:
            print(f"Processing t={time_v}")
            self._process_time_step(X, Y, Z, time_v)
        print(f"Outputs saved to {self.cfg.output_dir}")

    def _process_time_step(self, X, Y, Z, time_v):
        """
        处理单个时间步的数据，包括切片、涡旋检测和绘图。
        """
        # 切片处理：取 z=0.135 的数据
        slice_z = 0.135
        mask = np.isclose(Z, slice_z, atol=1e-6)  # 找到接近 z=0.135 的数据
        if not mask.any():
            nearest_z = Z[np.argmin(np.abs(Z - slice_z))]
            raise ValueError(f"No data at Z={slice_z}. Nearest Z={nearest_z:.6f}")
        X, Y = X[mask], Y[mask]


        Ua = fluidfoam.readvector(self.cfg.sol, str(time_v), "U.a")
        alpha = fluidfoam.readscalar(self.cfg.sol, str(time_v), "alpha.a")
        beta = fluidfoam.readscalar(self.cfg.sol, str(time_v), "alpha.b")
        gradU = fluidfoam.readtensor(self.cfg.sol, str(time_v), "grad(U.a)")
        vort = fluidfoam.readvector(self.cfg.sol, str(time_v), "vorticity")
        grad_beta = fluidfoam.readvector(self.cfg.sol, str(time_v), "grad(alpha.b)")
        grad_vort = fluidfoam.readtensor(self.cfg.sol, str(time_v), "grad(vorticity)")

                # 对切片后的数据进行掩码处理
        # 对切片后的数据进行掩码处理
        # X, Y, Z = X[mask], Y[mask], Z[mask]
        Ua = Ua[:, mask]  # 对三维向量场进行切片
        alpha = alpha[mask]
        beta = beta[mask]
        vort = vort[:, mask]
        grad_beta = grad_beta[:, mask]
        grad_vort = grad_vort[:, mask]

        head_x = self._find_head(X, Y, alpha)
        if head_x is None:
            print(f"Head not found at t={time_v}, skip.")
            return

        df = self._integrate_columns(X, Y, Ua, alpha, head_x, time_v)
        fields = self._build_fields(X, Y, Ua, alpha, beta, gradU, vort, grad_beta, grad_vort, df)
        xi, yi = fields["grid"]
        alpha_i = fields["alpha_i"]
        u_rot = fields["u_rot"]
        v_rot = fields["v_rot"]

        markers = {
            "$1/4H_0$": head_x - 0.25 * self.cfg.head_height,
            "$1/3H_0$": head_x - 0.33 * self.cfg.head_height,
            "$1/2H_0$": head_x - 0.5 * self.cfg.head_height,
            "$H_0$": head_x - self.cfg.head_height,
        }

        self._plot_vorticity(xi, yi, u_rot, v_rot, alpha_i, markers, time_v)
        vortices_q = self._detect_q_vortices(xi, yi, u_rot, v_rot, alpha_i, time_v)
        vortices_l2 = self._detect_lambda_vortices(xi, yi, u_rot, v_rot, alpha_i, time_v)

        self._plot_vortex_boundaries(xi, yi, vortices_q, u_rot, v_rot, alpha_i, markers, time_v)
        self._plot_lambda_boundaries(xi, yi, vortices_l2, u_rot, v_rot, alpha_i, markers, time_v)
        self._plot_lambda_mask(xi, yi, u_rot, v_rot, alpha_i, time_v)

        self._plot_streamlines(xi, yi, u_rot, v_rot, alpha_i, markers, time_v,
                               "Rotation Velocity Streamlines",
                               f"Rotation_streamlines_t{time_v}s.png")
        self._plot_streamlines(xi, yi, fields["uxi"], v_rot, alpha_i, markers, time_v,
                               "Original Velocity Streamlines",
                               f"Original_streamlines_t{time_v}s.png")


    def _find_head(self, X, Y, alpha):
        head = None
        for x in np.unique(X):
            if np.any((X == x) & (Y >= self.cfg.y_min) & (alpha > self.cfg.alpha_threshold)):
                head = x
        return head

    def _integrate_columns(self, X, Y, U, alpha, head_x, time_v):
        xs = np.unique(X[(X <= head_x) & (X >= 0.0)])
        rows = []
        for x in xs:
            mask = (X == x) & (Y >= 0.0) & (alpha > 1e-5)
            if not mask.any():
                continue
            metrics = self.integrate_column(Y[mask], U[0][mask], np.maximum(alpha[mask], 0.0))
            rows.append({
                "Time": time_v,
                "x": x,
                "U": metrics["U"],
                "H": metrics["H"],
                "y_crossing": metrics["zero_y"],
                "U_alpha": metrics["U"],
                "H_alpha": metrics["H_alpha"],
                "ALPHA_alpha": metrics["ALPHA"],
                "y_crossing_alpha": metrics["zero_y"],
            })
        df = pd.DataFrame(rows)
        out = os.path.join(self.cfg.output_dir, f"integration_results_t{time_v}.csv")
        df.to_csv(out, index=False)
        return df

    def _build_fields(self, X, Y, Ua, alpha, beta, gradU, vort, grad_beta, grad_vort, df):
        nx, ny = self.cfg.interp_shape
        x_grid = np.linspace(X.min(), X.max(), nx)
        y_grid = np.linspace(Y.min(), Y.max(), ny)
        xi, yi = np.meshgrid(x_grid, y_grid)

        u_mean_map = dict(zip(df["x"], df["U"]))
        u_rot_scatter = Ua[0].copy()
        for x_val, mean_u in u_mean_map.items():
            mask = np.isclose(X, x_val, atol=1e-9)
            if mask.any():
                u_rot_scatter[mask] -= mean_u

        base_fields = {
            "uxi": Ua[0],
            "v_rot": Ua[1],
            "alpha_i": alpha,
            "omega_z": vort[2],
            "gradbeta_x": grad_beta[0],
            "gradvort_x": grad_vort[2],
            "beta": beta,
            "u_rot": u_rot_scatter,
        }
        interp = {k: griddata((X, Y), v, (xi, yi), method="linear") for k, v in base_fields.items()}

        interp.update({
            "grid": (xi, yi),
            "u_rot": np.nan_to_num(interp["u_rot"]),
        })
        return interp

    # ---------- 涡旋检测 ---------- #

    def _detect_q_vortices(self, xi, yi, ux, uy, alpha, time_v):
        dy, dx = yi[1, 0] - yi[0, 0], xi[0, 1] - xi[0, 0]
        dUx = np.gradient(ux, dx, axis=1)
        dUy = np.gradient(ux, dy, axis=0)
        dVx = np.gradient(uy, dx, axis=1)
        dVy = np.gradient(uy, dy, axis=0)
        Q = self.q_criterion(dUx, dUy, dVx, dVy)

        mask = (Q > self.cfg.q_threshold) & (alpha > self.cfg.alpha_threshold)
        structure = np.ones(self.cfg.closing_kernel, dtype=bool)
        final_mask = binary_closing(mask, structure=structure) & (Q > 0.05)
        labeled, n_vort = label(final_mask)

        vortices = []
        for idx in range(1, n_vort + 1):
            region = labeled == idx
            if region.sum() < 10:
                continue
            coords = np.column_stack((xi[region], yi[region]))
            vel = np.column_stack((ux[region], uy[region]))
            try:
                geo = ManualPCA().fit(coords).properties()
                kin = ManualPCA().fit(vel - vel.mean(axis=0)).properties()
            except Exception as exc:
                print(f"PCA Q idx={idx} failed: {exc}")
                continue
            vortices.append({
                "id": idx,
                "time": time_v,
                "center": geo["center"],
                "length": geo["length"],
                "width": geo["width"],
                "geo_axis": geo["major_axis"],
                "kin_axis": kin["major_axis"],
                "geo_angle": np.degrees(geo["angle"]),
                "kin_angle": np.degrees(kin["angle"]),
                "area": np.pi * geo["length"] * geo["width"] / 4,
                "max_Q": np.max(Q[region]),
                "original_id": idx,
            })

        if vortices:
            vortices.sort(key=lambda v: v["area"], reverse=True)
            for new_id, vortex in enumerate(vortices, start=1):
                vortex["id"] = new_id
            self._save_vortices(vortices, self.csv_q)
        return vortices

    def _detect_lambda_vortices(self, xi, yi, ux, uy, alpha, time_v):
        dy, dx = yi[1, 0] - yi[0, 0], xi[0, 1] - xi[0, 0]
        ux_f = gaussian_filter(ux, sigma=1.0)
        uy_f = gaussian_filter(uy, sigma=1.0)
        lambda2 = self.lambda2_field(ux_f, uy_f, dx, dy)

        mask = (lambda2 < self.cfg.lambda2_threshold) & (alpha > self.cfg.alpha_threshold)
        structure = np.ones(self.cfg.closing_kernel, dtype=bool)
        final_mask = binary_closing(mask, structure=structure) & (lambda2 < 0)
        labeled, n_vort = label(final_mask)

        data = []
        for idx in range(1, n_vort + 1):
            region = labeled == idx
            if region.sum() < 10:
                continue
            try:
                props = ManualPCA().fit(np.column_stack((xi[region], yi[region]))).properties()
            except Exception as exc:
                print(f"PCA λ2 idx={idx} failed: {exc}")
                continue
            area = np.pi * props["length"] * props["width"] / 4
            if area <= 1e-4:
                continue
            data.append({
                "old": idx,
                "area": area,
                "props": props,
                "mask": region,
            })

        data.sort(key=lambda item: item["area"], reverse=True)
        vortices = []
        for new_id, entry in enumerate(data, start=1):
            props = entry["props"]
            vortices.append({
                "id": new_id,
                "time": time_v,
                "center": props["center"],
                "length": props["length"],
                "width": props["width"],
                "angle": np.degrees(props["angle"]),
                "area": entry["area"],
                "max_lambda2": np.min(lambda2[entry["mask"]]),
                "major_axis": props["major_axis"],
                "original_id": entry["old"],
            })

        if vortices:
            self._save_vortices(vortices, self.csv_l2)
        return vortices

    @staticmethod
    def _save_vortices(vortices: List[Dict], csv_path: str):
        df = pd.DataFrame(vortices)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"{len(df)} vortices saved -> {csv_path}")

    # ---------- 可视化 ---------- #

    def _plot_streamlines(self, xi, yi, ux, uy, alpha, markers, time_v, title, filename):
        plt.figure(figsize=self.style.fig_size)
        plt.contourf(xi, yi, alpha, levels=np.linspace(0, 0.015, 128), cmap="gray_r", alpha=0.75)
        mask = alpha > self.cfg.alpha_threshold
        plt.streamplot(
            xi, yi,
            np.where(mask, ux, 0.0),
            np.where(mask, uy, 0.0),
            color="#0343df", linewidth=1, density=8, arrowsize=2, arrowstyle="->",
        )
        plt.contour(xi, yi, alpha, **self.style.alpha_contour)
        self._draw_markers(markers)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.xlim(*self.style.x_lim)
        plt.ylim(*self.style.y_lim)
        plt.title(f"{title} (t={time_v}s)")
        plt.savefig(os.path.join(self.cfg.output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_vorticity(self, xi, yi, ux, uy, alpha, markers, time_v):
        dy, dx = yi[1, 0] - yi[0, 0], xi[0, 1] - xi[0, 0]
        omega_z = np.gradient(uy, dx, axis=1) - np.gradient(ux, dy, axis=0)
        plt.figure(figsize=self.style.fig_size)
        plt.contourf(xi, yi, omega_z, levels=np.linspace(-5, 5, 41), cmap="bwr", alpha=0.5)
        mask = alpha > self.cfg.alpha_threshold
        plt.streamplot(
            xi, yi,
            np.where(mask, ux, 0.0),
            np.where(mask, uy, 0.0),
            color="#0343df", linewidth=1, density=5, arrowsize=2, arrowstyle="->",
        )
        plt.contour(xi, yi, alpha, **self.style.alpha_contour)
        self._draw_markers(markers)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.xlim(*self.style.x_lim)
        plt.ylim(*self.style.y_lim)
        plt.title(rf"Vorticity Field ($\hat U_s$, t={time_v}s)")
        plt.savefig(os.path.join(self.cfg.output_dir, f"Vorticity_t{time_v}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_vortex_boundaries(self, xi, yi, vortices, ux, uy, alpha, markers, time_v):
        if not vortices:
            return
        plt.figure(figsize=self.style.fig_size)
        # === 添加精细网格 ===
        # 设置网格线的密度
        x_ticks = np.arange(self.style.x_lim[0], self.style.x_lim[1] + 0.1, 0.1)  # 每0.1m一条竖线
        y_ticks = np.arange(self.style.y_lim[0], self.style.y_lim[1] + 0.05, 0.05)  # 每0.05m一条横线
        
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.grid(True, alpha=1, linestyle='-', linewidth=0.5, color='lightgray')
        plt.contourf(xi, yi, alpha, levels=np.linspace(0, 0.015, 128), cmap="gray_r", alpha=0.75)
        mask = alpha > self.cfg.alpha_threshold
        plt.streamplot(
            xi, yi,
            np.where(mask, ux, 0.0),
            np.where(mask, uy, 0.0),
            color="#0343df", linewidth=1, density=8, arrowsize=2, arrowstyle="->",
        )
        ax = plt.gca()
        for vortex in vortices:
            ell = Ellipse(
                xy=vortex["center"],
                width=vortex["length"],
                height=vortex["width"],
                angle=vortex["geo_angle"],
                edgecolor="r",
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            ax.add_patch(ell)
            self._draw_axis(ax, vortex["center"], vortex["geo_axis"], vortex["length"], "r-")
            self._draw_axis(ax, vortex["center"], vortex["kin_axis"], vortex["length"], "m--")
            self._annotate(ax, vortex["center"], vortex["id"])
        plt.contour(xi, yi, alpha, **self.style.alpha_contour)
        self._draw_markers(markers)

        legend_elements = [
            Line2D([0], [0], color="r", linestyle="--", lw=2, label="Geometric Ellipse"),
            Line2D([0], [0], color="r", linestyle="-", lw=1.5, label="Geometric Axis"),
            Line2D([0], [0], color="m", linestyle="--", lw=1.5, label="Kinematic Axis"),
            # Line2D([0], [0], marker="x", color="r", lw=0, label="Vortex Center", markersize=10),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.title(f"Vortex Boundaries (PCA, t={time_v}s)")
        plt.xlim(*self.style.x_lim)
        plt.ylim(*self.style.y_lim)
        plt.savefig(os.path.join(self.cfg.output_dir, f"vortex_pca_dimensions_t{time_v}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_lambda_boundaries(self, xi, yi, vortices, ux, uy, alpha, markers, time_v):
        if not vortices:
            return
        plt.figure(figsize=self.style.fig_size)
        plt.contourf(xi, yi, alpha, levels=np.linspace(0, 0.015, 128), cmap="gray_r", alpha=0.75)
        plt.grid(True, alpha=1, linestyle="-", linewidth=0.5, color="lightgray")
        mask = alpha > self.cfg.alpha_threshold
        plt.streamplot(
            xi, yi,
            np.where(mask, ux, 0.0),
            np.where(mask, uy, 0.0),
            color="#0343df", linewidth=1, density=8, arrowsize=2, arrowstyle="->",
        )
        ax = plt.gca()
        for vortex in vortices:
            ell = Ellipse(
                xy=vortex["center"],
                width=vortex["length"],
                height=vortex["width"],
                angle=vortex["angle"],
                edgecolor="r",
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            ax.add_patch(ell)
            self._draw_axis(ax, vortex["center"], vortex["major_axis"], vortex["length"], "r-")
            self._annotate(ax, vortex["center"], vortex["id"])
        plt.contour(xi, yi, alpha, **self.style.alpha_contour)
        self._draw_markers(markers)

        legend_elements = [
            Line2D([0], [0], color="r", linestyle="--", lw=2, label="Lambda2 Ellipse"),
            Line2D([0], [0], color="r", linestyle="-", lw=1.5, label="Major Axis"),
            # Line2D([0], [0], marker="x", color="r", lw=0, label="Vortex Center", markersize=10),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.title(f"Lambda2 Vortex Boundaries (t={time_v}s)")
        plt.xlim(*self.style.x_lim)
        plt.ylim(*self.style.y_lim)
        plt.savefig(os.path.join(self.cfg.output_dir, f"lambda2_vortex_boundaries_t{time_v}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_lambda_mask(self, xi, yi, ux, uy, alpha, time_v):
        dy, dx = yi[1, 0] - yi[0, 0], xi[0, 1] - xi[0, 0]
        lambda2 = self.lambda2_field(ux, uy, dx, dy)
        mask = lambda2 < self.cfg.lambda2_threshold
        plt.figure(figsize=self.style.fig_size)
        plt.contourf(xi, yi, mask, levels=[0, 0.5, 1], colors=["none", "red"], alpha=0.3)
        stream_mask = alpha > self.cfg.alpha_threshold
        plt.streamplot(
            xi, yi,
            np.where(stream_mask, ux, 0.0),
            np.where(stream_mask, uy, 0.0),
            color="#0343df", linewidth=1, density=8, arrowsize=2, arrowstyle="->",
        )
        plt.contour(xi, yi, alpha, **self.style.alpha_contour)
        plt.title(rf"Vortex Regions ($\lambda_2 < 0$, t={time_v}s)")
        plt.xlim(*self.style.x_lim)
        plt.ylim(*self.style.y_lim)
        plt.savefig(os.path.join(self.cfg.output_dir, f"vortex_lambda_mask_t{time_v}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- 绘图辅助 ---------- #

    def _draw_markers(self, markers, y_text=0.32):
        for label, x_pos in markers.items():
            plt.axvline(x=x_pos, color=self.style.marker_color, linestyle="dashdot", linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, label, fontsize=20, zorder=3, color=self.style.marker_color)

    @staticmethod
    def _draw_axis(ax, center, axis, length, style):
        p1 = center - 0.5 * length * axis
        p2 = center + 0.5 * length * axis
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=1.5, zorder=4)

    @staticmethod
    def _annotate(ax, center, vid):
        ax.scatter(*center, c="r", s=50, marker="x", zorder=5)
        ax.text(center[0], center[1], str(vid),
                color="k", fontsize=12, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                zorder=6)


if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()