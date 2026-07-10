import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Use Times New Roman for all plot text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Make math text use Times New Roman as well (mathtext custom)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


@dataclass
class TimeStepVelocity3D:
    """Container for one time-step fields in structured layout (nx, ny, nz)."""

    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_a: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    q: np.ndarray


class VelocityVectorUaxAnalyzer:
    def __init__(self):
        # OpenFOAM case directory and output directory.
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_2"
        # self.output_dir = "/home/amber/postpro/velocity_vector/tc3d_d23_0604_2"
        self.output_dir = "/home/amber/postpro/velocity_vector/tc3d_d09_0604_2"
        self.times = [ 2,5,7,12,15,20,25,35]

        # alpha.a threshold for current head detection.
        self.alpha_threshold = 1e-5
        # Characteristic height for non-dimensional x.
        self.H = 0.3
        # Velocity field name in OpenFOAM.
        self.velocity_field = "U.b"

        # Plot style.
        self.fig_size = (10, 4)
        self.cmap = "gray_r"
        self.vec_abs_percentile = 98
        self.interp_nx = 460
        self.interp_ny = 120
        self.quiver_step_x = 18
        self.quiver_step_y = 8
        self.velocity_ref = None  # None means each figure uses vec_abs_percentile of speed.
        self.vector_color = "#FF00FF"
        self.quiver_key_x = 0.69
        self.quiver_key_y = 0.90
        self.reference_arrow_inches = 0.32  # On-page length for |u| = U_ref.
        self.quiver_scale = 1.0 / self.reference_arrow_inches
        self.quiver_width = 0.0020
        self.arrow_scale = 0.85  # Overall scaling for arrowhead size
        self.alpha_contour_color = "0.25"
        self.alpha_contour_linewidth = 1.1
        self.q_contour_levels = [1.0]
        self.q_contour_linewidth = 1.0
        self.q_smooth_sigma = 1.0
        self.q_contour_color = "#FF00FF"

        # Font sizes.
        self.label_fontsize = 32
        self.tick_fontsize = 30
        self.cbar_labelsize = 32
        self.cbar_ticksize = 30

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
    def _compute_q_criterion_3d(grad_u: np.ndarray) -> np.ndarray:
        """Compute physical 3D Q-criterion from OpenFOAM grad(U) tensor."""
        if grad_u.shape[0] != 9:
            raise ValueError(f"Expected grad_u with 9 tensor components, got {grad_u.shape[0]}")

        gxx, gxy, gxz, gyx, gyy, gyz, gzx, gzy, gzz = grad_u
        s_xx = gxx
        s_yy = gyy
        s_zz = gzz
        s_xy = 0.5 * (gxy + gyx)
        s_xz = 0.5 * (gxz + gzx)
        s_yz = 0.5 * (gyz + gzy)

        w_xy = 0.5 * (gxy - gyx)
        w_xz = 0.5 * (gxz - gzx)
        w_yz = 0.5 * (gyz - gzy)

        strain_sq = (
            s_xx**2
            + s_yy**2
            + s_zz**2
            + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)
        )
        rot_sq = 2.0 * (w_xy**2 + w_xz**2 + w_yz**2)
        return 0.5 * (rot_sq - strain_sq)

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if valid_x.size == 0:
            return None
        return int(valid_x.max())

    def _load_velocity_3d(
        self, grid: Dict[str, np.ndarray], time_v: float
    ) -> Optional[TimeStepVelocity3D]:
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
            vel_raw = fluidfoam.readvector(self.sol, time_dir, self.velocity_field)
            vel_3d = self._reshape_sorted(vel_raw, sort_idx, nx, ny, nz)
            ux = vel_3d[0]
            uy= vel_3d[1]
        except Exception as exc:
            print(f"Read failed for {self.velocity_field} at t={time_v}: {exc}")
            return None

        try:
            grad_raw = fluidfoam.readtensor(self.sol, time_dir, f"grad({self.velocity_field})")
            grad_3d = self._reshape_sorted(grad_raw, sort_idx, nx, ny, nz)
            q_3d = self._compute_q_criterion_3d(grad_3d)
        except Exception as exc:
            print(f"Read failed for grad({self.velocity_field}) at t={time_v}: {exc}")
            return None

        return TimeStepVelocity3D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            alpha_a=alpha_a,
            ux=ux,
            uy=uy,
            q=q_3d,
        )

    @staticmethod
    def _interp_to_uniform_grid(
        x_old: np.ndarray,
        y_old: np.ndarray,
        fields: Dict[str, np.ndarray],
        nx_new: int,
        ny_new: int,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        x_new = np.linspace(float(np.min(x_old)), float(np.max(x_old)), nx_new)
        y_new = np.linspace(float(np.min(y_old)), float(np.max(y_old)), ny_new)

        fields_y = {}
        for name, field in fields.items():
            interp_y = np.full((x_old.size, y_new.size), np.nan, dtype=float)
            for i in range(x_old.size):
                row = field[i, :]
                mask = np.isfinite(row)
                if np.count_nonzero(mask) >= 2:
                    interp_y[i, :] = np.interp(
                        y_new, y_old[mask], row[mask], left=np.nan, right=np.nan
                    )
            fields_y[name] = interp_y

        fields_new = {}
        for name, field in fields_y.items():
            interp_xy = np.full((x_new.size, y_new.size), np.nan, dtype=float)
            for j in range(y_new.size):
                col = field[:, j]
                mask = np.isfinite(col)
                if np.count_nonzero(mask) >= 2:
                    interp_xy[:, j] = np.interp(
                        x_new, x_old[mask], col[mask], left=np.nan, right=np.nan
                    )
            fields_new[name] = interp_xy

        return x_new, y_new, fields_new

    def _smooth_masked_field(self, field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Gaussian smooth a field using only values inside mask."""
        valid = mask & np.isfinite(field)
        if self.q_smooth_sigma <= 0.0 or not np.any(valid):
            return np.where(valid, field, np.nan)

        weights = gaussian_filter(valid.astype(float), sigma=self.q_smooth_sigma)
        values = gaussian_filter(
            np.where(valid, field, 0.0),
            sigma=self.q_smooth_sigma,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = values / weights
        smoothed[weights < 1e-6] = np.nan
        return np.where(mask, smoothed, np.nan)

    def _save_uax_vector(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        ux_2d: np.ndarray,
        uy_2d: np.ndarray,
        alpha_2d: np.ndarray,
        q_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ) -> None:
        out_dir = os.path.join(self.output_dir, f"vector_uax_t{time_v:.2f}")
        os.makedirs(out_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_dime = (head_x - x_seg) / self.H
        y_vals = y_axis / self.H

        # Velocity components in the transformed coordinates:
        # x* = (x_front - x) / H, y* = y / H.
        # The x component changes sign, while the vertical component does not.
        uax_seg = -ux_2d[: head_idx + 1, :]
        uay_seg = uy_2d[: head_idx + 1, :]
        a_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)
        q_seg = q_2d[: head_idx + 1, :]

        x_order = np.argsort(x_dime)
        x_plot = x_dime[x_order]
        uax_plot = uax_seg[x_order, :]
        uay_plot = uay_seg[x_order, :]
        a_plot = a_seg[x_order, :]
        q_plot = q_seg[x_order, :]

        if x_plot.size < 2 or np.any(np.diff(x_plot) <= 0):
            print(f"Non-monotonic x grid at t={time_v}. Skip vector output.")
            return

        if y_vals.size < 2 or np.any(np.diff(y_vals) <= 0):
            print(f"Non-monotonic y grid at t={time_v}. Skip vector output.")
            return

        x_plot, y_vals, interp = self._interp_to_uniform_grid(
            x_plot,
            y_vals,
            {"u": uax_plot, "v": uay_plot, "alpha": a_plot, "q": q_plot},
            self.interp_nx,
            self.interp_ny,
        )
        uax_plot = interp["u"]
        uay_plot = interp["v"]
        a_plot = interp["alpha"]
        q_plot = interp["q"]

        xx, yy = np.meshgrid(x_plot, y_vals, indexing="ij")
        valid = a_plot[np.isfinite(a_plot)]
        if valid.size == 0 or float(np.nanmax(valid)) <= 0.0:
            print(f"Invalid alpha field at t={time_v}. Skip plotting.")
            return

        alpha_plot = np.clip(a_plot, 0.0, 0.01)

        # Downsample arrows for readability.
        sx = slice(None, None, self.quiver_step_x)
        sy = slice(None, None, self.quiver_step_y)
        xx_q = xx[sx, sy]
        yy_q = yy[sx, sy]
        u_q = uax_plot[sx, sy]
        v_q = uay_plot[sx, sy]
        alpha_q = alpha_plot[sx, sy]
        current_mask_q = alpha_q > self.alpha_threshold

        fig_vec, ax_vec = plt.subplots(figsize=self.fig_size)
        fig_vec.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.80)
        ax_vec.contourf(
            xx,
            yy,
            alpha_plot,
            levels=np.linspace(0, 0.01, 121),
            cmap=self.cmap,
            extend="neither",
        )
        ax_vec.contour(
            xx,
            yy,
            alpha_plot,
            levels=[self.alpha_threshold],
            colors=self.alpha_contour_color,
            linestyles="--",
            linewidths=self.alpha_contour_linewidth,
        )

        speed_q = np.sqrt(u_q**2 + v_q**2)
        finite_speed = speed_q[current_mask_q & np.isfinite(speed_q)]
        if finite_speed.size == 0:
            print(f"No valid velocity vectors inside alpha>{self.alpha_threshold} at t={time_v}.")
            plt.close(fig_vec)
            return

        if self.velocity_ref is None:
            u_ref = float(np.nanpercentile(finite_speed, self.vec_abs_percentile))
        else:
            u_ref = float(self.velocity_ref)
        if not np.isfinite(u_ref) or u_ref <= 0.0:
            u_ref = float(np.nanmax(finite_speed))
        if not np.isfinite(u_ref) or u_ref <= 0.0:
            print(f"Degenerated velocity range at t={time_v}. Skip vector plotting.")
            plt.close(fig_vec)
            return

        u_nd = u_q / u_ref
        v_nd = v_q / u_ref
        u_arrow = np.where(current_mask_q, u_nd, np.nan)
        v_arrow = np.where(current_mask_q, v_nd, np.nan)

        # Dimensionless Q from precomputed q_plot (physical Q): Q* = Q * (H / U_ref)^2
        q_nd_full = q_plot * (self.H / u_ref) ** 2

        mask_alpha = alpha_plot > self.alpha_threshold
        q_contour_field = self._smooth_masked_field(q_nd_full, mask_alpha)

        q = ax_vec.quiver(
            xx_q,
            yy_q,
            u_arrow,
            v_arrow,
            color=self.vector_color,
            angles="xy",
            scale_units="inches",
            scale=self.quiver_scale,
            width=self.quiver_width,  # 箭头宽度
            headwidth=3.0 * self.arrow_scale,  # 头部宽度
            headlength=4.0 * self.arrow_scale, # 头部长度
            headaxislength=3.5 * self.arrow_scale,
            pivot="mid",
        )
        ax_vec.quiverkey(
            q,
            X=self.quiver_key_x,
            Y=self.quiver_key_y,
            U=1.0,
            label=rf"$|\mathbf{{u}}_\mathrm{{s}}|={u_ref:.3f}\ \mathrm{{m/s}}$",
            labelpos="E",
            coordinates="axes",
            color=self.vector_color,
            labelcolor="black",
            fontproperties={"size": self.tick_fontsize},
        )
        ax_vec.set_xlabel(r"$(x_{\mathrm{front}}-x)/H_0$", fontsize=self.label_fontsize)
        ax_vec.set_yticks([0.0, 0.5, 1.0])
        ax_vec.set_ylabel(r"$z*$", fontsize=self.label_fontsize, rotation=0, labelpad=20)
        ax_vec.set_xlim(float(np.max(x_plot)), 0.0)
        ax_vec.set_ylim(0.0, 1.0)
        ax_vec.tick_params(axis="both", labelsize=self.tick_fontsize, width=0.8, length=3.5)
        for spine in ax_vec.spines.values():
            spine.set_linewidth(0.8)

        out_vector_png = os.path.join(out_dir, f"vector_uax_spanwise_t{time_v:.2f}.png")
        fig_vec.savefig(out_vector_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_vec)
        print(f"Saved Vector Figure: {out_vector_png}")

        fig_q, ax_q = plt.subplots(figsize=self.fig_size)
        fig_q.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.80)
        ax_q.contourf(
            xx,
            yy,
            alpha_plot,
            levels=np.linspace(0, 0.01, 121),
            cmap=self.cmap,
            extend="neither",
        )
        ax_q.contour(
            xx,
            yy,
            alpha_plot,
            levels=[self.alpha_threshold],
            colors=self.alpha_contour_color,
            linestyles="--",
            linewidths=self.alpha_contour_linewidth,
        )
        q_valid = q_contour_field[np.isfinite(q_contour_field)]
        if q_valid.size > 0:
            q_min = float(np.nanmin(q_valid))
            q_max = float(np.nanmax(q_valid))
            q_levels = [level for level in self.q_contour_levels if q_min <= level <= q_max]
            if not q_levels:
                # fallback to adaptive levels across data range
                if q_max > q_min:
                    q_levels = list(np.linspace(q_min, q_max, 6))
                else:
                    q_levels = [q_min]
            cs = ax_q.contour(
                xx, yy, q_contour_field,
                levels=q_levels,
                colors=self.q_contour_color,
                linewidths=self.q_contour_linewidth,
                linestyles='-',
            )
        ax_q.set_xlabel(r"$(x_{\mathrm{front}}-x)/H_0$", fontsize=self.label_fontsize)
        ax_q.set_yticks([0.0, 0.5, 1.0])
        ax_q.set_ylabel(r"$z*$", fontsize=self.label_fontsize, rotation=0, labelpad=20)
        ax_q.set_xlim(float(np.max(x_plot)), 0.0)
        ax_q.set_ylim(0.0, 1.0)
        ax_q.tick_params(axis="both", labelsize=self.tick_fontsize, width=0.8, length=3.5)
        for spine in ax_q.spines.values():
            spine.set_linewidth(0.8)

        out_q_png = os.path.join(out_dir, f"q_contour_spanwise_t{time_v:.2f}.png")
        fig_q.savefig(out_q_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig_q)
        print(f"Saved Q Figure: {out_q_png}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> None:
        data_3d = self._load_velocity_3d(grid, float(time_v))
        if data_3d is None:
            return

        alpha_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_2d)
        if head_idx is None:
            print(
                f"No alpha.a > threshold ({self.alpha_threshold}) at t={time_v}. Skip output."
            )
            return

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx})")

        ux_2d = np.mean(data_3d.ux, axis=2)
        uy_2d = np.mean(data_3d.uy, axis=2)
        q_2d = np.mean(data_3d.q, axis=2)

        self._save_uax_vector(
            float(time_v),
            data_3d.x_axis,
            data_3d.y_axis,
            ux_2d,
            uy_2d,
            alpha_2d,
            q_2d,
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
    analyzer = VelocityVectorUaxAnalyzer()
    analyzer.run_analysis()
