import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Use Times New Roman for all plot text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Make math text use Times New Roman as well (mathtext custom)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


@dataclass
class TimeStepLambda23D:
    """Container for one time-step fields in structured layout (nx, ny, nz)."""

    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_a: np.ndarray
    lambda2: np.ndarray


class Lambda2SpanwiseAnalyzer:
    def __init__(self):
        # OpenFOAM case directory and output directory.
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_12"
        self.output_dir = "/home/amber/postpro/velocity_lambda2/tc3d_d09_0327_12"
        self.times = [0.5, 2, 5, 15]

        # alpha.a threshold for current head detection.
        self.alpha_threshold = 1e-5
        # Characteristic height for non-dimensional x.
        self.H = 0.3
        # Velocity field name in OpenFOAM.
        self.velocity_field = "U.b"

        # Plot style.
        self.fig_size = (20, 10)
        self.alpha_cmap = "gray_r"
        self.lambda2_cmap = "Blues_r"
        self.lambda2_alpha = 0.62
        self.lambda2_abs_percentile = 98.0
        self.lambda2_levels = 121
        self.alpha_contour_levels = [0.0025, 0.005, 0.0075, 0.01]

        # Font sizes.
        self.label_fontsize = 52
        self.tick_fontsize = 50
        self.cbar_labelsize = 52
        self.cbar_ticksize = 50

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _build_grid_cache(
        X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

        x3d = X_raw[sort_idx].reshape((nx, ny, nz), order="C")[:, 0, 0]
        y3d = Y_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, :, 0]
        z3d = Z_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, 0, :]

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis_3d": x3d,
            "y_axis_3d": y3d,
            "z_axis_3d": z3d,
        }

    @staticmethod
    def _reshape_sorted(
        field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int
    ) -> np.ndarray:
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    @staticmethod
    def _compute_lambda2(grad_u: np.ndarray) -> np.ndarray:
        """Compute lambda2 from grad(U) using the openfoam tensor ordering."""
        if grad_u.shape[0] != 9:
            raise ValueError(f"Expected grad_u with 9 tensor components, got {grad_u.shape[0]}")

        gxx, gxy, gxz, gyx, gyy, gyz, gzx, gzy, gzz = grad_u
        g = np.empty(grad_u.shape[1:] + (3, 3), dtype=float)
        g[..., 0, 0] = gxx
        g[..., 0, 1] = gxy
        g[..., 0, 2] = gxz
        g[..., 1, 0] = gyx
        g[..., 1, 1] = gyy
        g[..., 1, 2] = gyz
        g[..., 2, 0] = gzx
        g[..., 2, 1] = gzy
        g[..., 2, 2] = gzz

        s = 0.5 * (g + np.swapaxes(g, -1, -2))
        omega = 0.5 * (g - np.swapaxes(g, -1, -2))
        m = np.matmul(s, s) + np.matmul(omega, omega)
        eigvals = np.linalg.eigvalsh(m)
        return eigvals[..., 1]

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if valid_x.size == 0:
            return None
        return int(valid_x.max())

    def _load_fields_3d(
        self, grid: Dict[str, np.ndarray], time_v: float
    ) -> Optional[TimeStepLambda23D]:
        print(f"\n>>> Processing time: {time_v}")
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        try:
            alpha_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
            alpha_a = self._reshape_sorted(alpha_raw, sort_idx, nx, ny, nz)
        except Exception as exc:
            print(f"Read failed for alpha.a at t={time_v}: {exc}")
            return None

        try:
            grad_raw = fluidfoam.readtensor(self.sol, time_dir, f"grad({self.velocity_field})")
            grad_3d = self._reshape_sorted(grad_raw, sort_idx, nx, ny, nz)
            lambda2 = self._compute_lambda2(grad_3d)
        except Exception as exc:
            print(f"Read failed for grad({self.velocity_field}) at t={time_v}: {exc}")
            return None

        return TimeStepLambda23D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            alpha_a=alpha_a,
            lambda2=lambda2,
        )

    def _save_lambda2_figure(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        alpha_2d: np.ndarray,
        lambda2_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ) -> None:
        out_dir = os.path.join(self.output_dir, f"lambda2_t{time_v:.2f}")
        os.makedirs(out_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_dime = (head_x - x_seg) / self.H
        y_vals = y_axis / self.H

        x_order = np.argsort(x_dime)
        x_plot = x_dime[x_order]
        alpha_plot = np.maximum(alpha_2d[: head_idx + 1, :][x_order, :], 0.0)
        lambda2_plot = lambda2_2d[: head_idx + 1, :][x_order, :]

        if x_plot.size < 2 or np.any(np.diff(x_plot) <= 0):
            print(f"Non-monotonic x grid at t={time_v}. Skip lambda2 output.")
            return

        if y_vals.size < 2 or np.any(np.diff(y_vals) <= 0):
            print(f"Non-monotonic y grid at t={time_v}. Skip lambda2 output.")
            return

        xx, yy = np.meshgrid(x_plot, y_vals, indexing="ij")
        alpha_valid = alpha_plot[np.isfinite(alpha_plot)]
        lambda2_valid = lambda2_plot[np.isfinite(lambda2_plot)]
        if alpha_valid.size == 0 or float(np.nanmax(alpha_valid)) <= 0.0:
            print(f"Invalid alpha field at t={time_v}. Skip plotting.")
            return
        if lambda2_valid.size == 0:
            print(f"Invalid lambda2 field at t={time_v}. Skip plotting.")
            return

        alpha_bg = np.clip(alpha_plot, 0.0, 0.01)
        lambda2_neg_valid = -lambda2_valid[lambda2_valid < 0.0]
        if lambda2_neg_valid.size == 0:
            print(f"No negative lambda2 vortex-core region at t={time_v}. Skip plotting.")
            return
        lambda2_abs_max = float(
            np.nanpercentile(lambda2_neg_valid, self.lambda2_abs_percentile)
        )
        if not np.isfinite(lambda2_abs_max) or lambda2_abs_max <= 0.0:
            lambda2_abs_max = float(np.nanmax(lambda2_neg_valid))
        if not np.isfinite(lambda2_abs_max) or lambda2_abs_max <= 0.0:
            print(f"Degenerated negative lambda2 range at t={time_v}. Skip plotting.")
            return

        norm = Normalize(vmin=-lambda2_abs_max, vmax=0.0)
        lambda2_core = np.ma.masked_where(lambda2_plot >= 0.0, lambda2_plot)

        fig, ax = plt.subplots(figsize=self.fig_size)
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.80)

        cf_bg = ax.contourf(
            xx,
            yy,
            alpha_bg,
            levels=np.linspace(0, 0.01, 121),
            cmap=self.alpha_cmap,
            extend="neither",
        )
        alpha_levels = [
            level for level in self.alpha_contour_levels if level < float(np.nanmax(alpha_plot))
        ]
        if alpha_levels:
            ax.contour(
                xx,
                yy,
                alpha_plot,
                levels=alpha_levels,
                colors="0.35",
                linewidths=0.9,
                alpha=0.75,
            )

        cf_lambda2 = ax.contourf(
            xx,
            yy,
            lambda2_core,
            levels=np.linspace(-lambda2_abs_max, 0.0, self.lambda2_levels),
            cmap=self.lambda2_cmap,
            norm=norm,
            alpha=self.lambda2_alpha,
            extend="min",
        )
        ax.contour(xx, yy, lambda2_plot, levels=[0.0], colors="navy", linewidths=1.2)

        cbar_lambda2 = fig.colorbar(
            cf_lambda2,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            fraction=0.045,
            ticks=[-lambda2_abs_max, -0.5 * lambda2_abs_max, 0.0],
        )
        cbar_lambda2.set_label(r"$\lambda_2<0$", fontsize=self.cbar_labelsize)
        cbar_lambda2.ax.tick_params(labelsize=self.cbar_ticksize)

        ax.set_xlabel(r"$(x_{\mathrm{front}}-x)/H_0$", fontsize=self.label_fontsize)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel(r"$\tilde{z}$", fontsize=self.label_fontsize, rotation=0, labelpad=20)
        ax.set_xlim(float(np.max(x_plot)), float(np.min(x_plot)))
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)

        out_png = os.path.join(out_dir, f"lambda2_spanwise_t{time_v:.2f}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved Figure: {out_png}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> None:
        data_3d = self._load_fields_3d(grid, float(time_v))
        if data_3d is None:
            return

        alpha_2d = np.mean(data_3d.alpha_a, axis=2)
        lambda2_2d = np.mean(data_3d.lambda2, axis=2)

        head_idx = self._locate_head_index(alpha_2d)
        if head_idx is None:
            print(
                f"No alpha.a > threshold ({self.alpha_threshold}) at t={time_v}. Skip output."
            )
            return

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx})")

        self._save_lambda2_figure(
            float(time_v),
            data_3d.x_axis,
            data_3d.y_axis,
            alpha_2d,
            lambda2_2d,
            head_idx,
            head_x,
        )

    def run_analysis(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        for t in self.times:
            self.process_time_step(grid, float(t))


if __name__ == "__main__":
    analyzer = Lambda2SpanwiseAnalyzer()
    analyzer.run_analysis()
