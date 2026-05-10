import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


@dataclass
class TimeStepTerms3D:
    """Container for one time-step fields used in postprocessing only.

    All arrays are in the reconstructed structured-grid layout (nx, ny, nz),
    not the original flattened OpenFOAM ordering.
    """

    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_a: np.ndarray
    Ub: np.ndarray
    terms: Dict[str, np.ndarray]
    kb: np.ndarray
    ubvorticity_x: Optional[np.ndarray] = None
    ubvorticity_y: Optional[np.ndarray] = None
    ubvorticity_z: Optional[np.ndarray] = None
    gradUb_ux_dz: Optional[np.ndarray] = None
    gradMixrho_z: Optional[np.ndarray] = None
    lambda2: Optional[np.ndarray] = None


class TKEBudgetAnalyzer:
    def __init__(self):
        # OpenFOAM case directory and output directory for generated CSV/figures.
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230327_1"
        # self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0327_1"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_11"
        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d09_0327_1"
        # Physical times to be processed.
        self.times = [15,25,35]

        # Threshold to detect current head position from alpha.a.
        self.alpha_threshold = 1e-5
        # Default style settings shared by all figures.
        self.fig_size = (20, 3)
        self.curve_lw = 2.0
        self.x_dime_max = 4.0
        self.cloud_fig_size = (9, 3.2)
        self.cloud_levels = 121
        self.cloud_percentile = (1.0, 99.0)
        # If True, VTK scalar values are clipped using the same percentile range as cloud plots.
        self.vtk_match_cloud_percentile = True
        self.save_curve_png = True
        self.save_comparison_png = True
        self.save_vtk = True
        # Variables compared across all selected times in one summary figure set.
        # You can use either raw names (e.g. "G") or curve-column names (e.g. "G_avg").
        self.comparison_columns = ["G", "convection", "diff", "drag1", "dissipation", "ratio"]
        # Set None to use all variables from comparison_columns.
        # Set an integer (e.g. 2) to only output the first N variables.
        self.num_comparison_variables = None
        # Water density used to scale all output budget curves.
        # self.water_density = 1000.0
        self.U = 0.26
        self.H = 0.3
        # G term uses beta from 3D field: beta = 1 - alpha.a.
        

        # Mapping: output column name -> OpenFOAM field name.
        # Keep/edit this list based on what has already been computed in OF.
        self.of_term_fields = {
            "convection": "Kconvection",
            "G": "Kprod",
            "density_gradient": "Kgrad",
            "dissipation": "Kdissip",
            "drag1": "drag1",
            "drag2": "drag2",
            "drag3": "drag3",
            "dkdtof": "dkdt",
            "diff": "Kdiff1",
            "ksource": "Ksource",
            "kresidual": "Kresidual",
           
            
        }

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        """Format float time to match OpenFOAM folder names like 16, 15.5, 16.25.

        OpenFOAM time folders are usually compact (no trailing zeros), so
        `:g` formatting keeps compatibility with folder naming.
        """
        return f"{float(time_v):g}"

    @staticmethod
    def _nondim_time(time_v: float) -> float:
        """Convert physical time to the nondimensional time used for outputs."""
        return float(time_v) * 0.85

    def _time_tag(self, time_v: float) -> str:
        """Format nondimensional time for filenames and directory names."""
        return f"{time_v:.2f}"

    def _time_label(self, time_v: float) -> str:
        """Format nondimensional time for plot titles and log messages."""
        return rf"$t*={self._nondim_time(time_v):.2f}$"

    @staticmethod
    def _build_grid_cache(X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Precompute mesh metadata used to reshape flattened OpenFOAM fields.

        OpenFOAM read functions return flattened arrays. We lexicographically sort
        by (X, Y, Z), then reshape to consistent (nx, ny, nz) structured arrays.
        """
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        # Sort by x first, then y, then z for deterministic reconstruction.
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
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        """Reshape flattened OpenFOAM data after lexsort; reconstruction order is C.

        Supports scalar fields with shape (N,) and vector-like fields with shape
        (ncomp, N). In this script we mainly use scalar fields.
        """
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    @staticmethod
    def _compute_lambda2(grad_u: np.ndarray) -> np.ndarray:
        """Compute the lambda2 vortex-identification scalar from grad(U).

        The input is expected to have shape (9, nx, ny, nz) and use the same
        component ordering as your OpenFOAM tensor field: the first three
        entries are the x-derivative row [dudx, dvdx, dwdx], the next three are
        the y-derivative row [dudy, dvdy, dwdy], and the last three are the
        z-derivative row [dudz, dvdz, dwdz]. The output is the second-largest
        eigenvalue of S^2 + Omega^2 at each cell.
        """
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

    @staticmethod
    def _vertical_integral(field2d: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Vertical integral along y for each x (no normalization).

        field2d shape is (nx, ny). Integration is along y (axis=1).
        """
        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(field2d, x=y_coords, axis=1)
        else:
            integral = np.trapz(field2d, x=y_coords, axis=1)

       
        return integral
    

    @staticmethod
    def _vertical_average_to_zerocity_zero(
        field2d: np.ndarray,
        y_coords: np.ndarray,
        ubx2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average along y from y>=0.01 to first Ubx positive-to-nonpositive crossing.

        If no crossing is found, use y max as the upper bound.
        """
        nx = field2d.shape[0]
        y_lower = 0.001
        out = np.zeros(nx, dtype=float)
        heights = np.zeros(nx, dtype=float)

        for i in range(nx):
            f_profile = field2d[i]
            u_profile = ubx2d[i]
            valid = np.isfinite(f_profile) & np.isfinite(u_profile) & np.isfinite(y_coords)
            

            y_valid = y_coords[valid]
            f_valid = f_profile[valid]
            u_valid = u_profile[valid]

            zero_y = None
            for j in range(len(u_valid) - 1):
                if y_valid[j] < y_lower:
                    continue
                if u_valid[j] > 0.0 and u_valid[j + 1] <= 0.0:
                    zero_y = float(y_valid[j + 1])
                    break

            y_upper = float(y_valid[-1]) if zero_y is None else float(zero_y)
            

            active_mask = (y_valid >= y_lower) & (y_valid <= y_upper)
            y_sel = y_valid[active_mask]
            f_sel = f_valid[active_mask]

           
            if y_sel.size < 2:
                        heights[i] = 0.0
                        out[i] = 0.0
                        continue

            if hasattr(np, "trapezoid"):
                numerator = float(np.trapezoid(f_sel, x=y_sel))
            else:
                numerator = float(np.trapz(f_sel, x=y_sel))

            height = float(y_sel[-1] - y_sel[0])
            heights[i] = height
            if height > 1e-12:
                out[i] = numerator / height
            else:
                out[i] = 0.0


        return out, heights

    def _trim_x_dime(self, x_seg: np.ndarray, x_head: float):
        """Trim x-axis segment so x_dime stays within [0, x_dime_max]."""
        x_dime = (x_head - x_seg) / 0.3
        mask = (x_dime >= 0.0) & (x_dime <= self.x_dime_max)
        return x_seg[mask], x_dime[mask], mask

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        """Find last x index where spanwise-averaged alpha_a exceeds threshold at any y.

        This x index is used as the front/head location; all curves are truncated
        to [0, head_idx] so only the current body/head region is plotted.
        """
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if len(valid_x) == 0:
            return None
        return int(valid_x.max())

    def _load_terms_3d(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepTerms3D]:
        """Load OF-precomputed budget terms for one time.

        Important: this script does not derive terms via gradients; it only reads
        fields that were already computed and written by OpenFOAM.
        """
        time_tag = self._time_tag(time_v)
        time_label = self._time_label(time_v)
        print(f"\n>>> Processing time: {time_label}")
        time_dir = self._time_to_dir_name(time_v)

        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        try:
            alpha_a_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
            ub_raw = fluidfoam.readvector(self.sol, time_dir, "U.b")
            ubvorticity = fluidfoam.readvector(self.sol, time_dir, "vorticity_Ub")
            gradUb_raw = fluidfoam.readtensor(self.sol, time_dir, "grad(U.b)")
            gradMixrho_raw = fluidfoam.readvector(self.sol, time_dir, "gradMixedrho")
            k_raw = fluidfoam.readscalar(self.sol, time_dir, "k.b")
        except Exception as exc:
            print(f"Read failed for alpha.a or U.b at {time_tag}: {exc}")
            return None

        alpha_a = self._reshape_sorted(alpha_a_raw, sort_idx, nx, ny, nz)
        Ub = self._reshape_sorted(ub_raw, sort_idx, nx, ny, nz)
        gradUb = self._reshape_sorted(gradUb_raw, sort_idx, nx, ny, nz)
        gradMixrho = self._reshape_sorted(gradMixrho_raw, sort_idx, nx, ny, nz)
        kb = self._reshape_sorted(k_raw, sort_idx, nx, ny, nz)
        # Keep user's convention: use y-direction components for Rig evaluation.
        gradUb_ux_dz = gradUb[3]
        gradMixrho_z = gradMixrho[1]
        lambda2 = self._compute_lambda2(gradUb)
        ubvorticity = self._reshape_sorted(ubvorticity, sort_idx, nx, ny, nz)
        ubvorticity_x = ubvorticity[0]
        ubvorticity_y = ubvorticity[1]
        ubvorticity_z = ubvorticity[2]
        
        

        # Read each configured budget term if the field exists at this time.
        loaded_terms: Dict[str, np.ndarray] = {}
        for out_name, of_name in self.of_term_fields.items():
            try:
                term_raw = fluidfoam.readscalar(self.sol, time_dir, of_name)
                loaded_terms[out_name] = self._reshape_sorted(term_raw, sort_idx, nx, ny, nz)
            except Exception as exc:
                print(f"Skip missing term '{of_name}' at {time_tag}: {exc}")

        if not loaded_terms:
            print(f"No budget terms were loaded at {time_tag}. Skip output.")
            return None

        return TimeStepTerms3D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            alpha_a=alpha_a,
            Ub=Ub,  
            terms=loaded_terms,
            ubvorticity_x=ubvorticity_x,
            ubvorticity_y=ubvorticity_y,
            ubvorticity_z=ubvorticity_z,
            gradUb_ux_dz=gradUb_ux_dz,
            gradMixrho_z=gradMixrho_z,
            lambda2=lambda2,
            kb=kb,
        )

    @staticmethod
    def _spanwise_average_terms(terms_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Average each term along z to obtain unified 2D terms (x, y)."""
        return {name: np.mean(field, axis=2) for name, field in terms_3d.items()}

    def _average_to_curves(
        self,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        ubx_2d: np.ndarray,
        head_idx: int,
        kb_2d: np.ndarray,
    ) -> pd.DataFrame:
        # Keep only data up to the detected head location.
        x_seg = x_axis[: head_idx + 1]
        x_head = x_axis[head_idx]
        x_dime = (x_head - x_seg) / 0.3

        curves = {
            "x": x_seg,
            "x_dime": x_dime,
        }

        for name, field in terms_2d.items():
            field_seg = field[: head_idx + 1, :]

            # 1) Velocity-zero-based vertical average: integrate from the bottom up to Ubx=0.
            curve_avg, curve_height = self._vertical_average_to_zerocity_zero(field_seg, y_axis, ubx_2d[: head_idx + 1, :])
            curves[f"{name}_avg"] = curve_avg
            curves[f"{name}_height"] = curve_height

            # 2) Full-depth vertical integral with no denominator.
            curve_in = self._vertical_integral(field_seg, y_axis)
            curves[f"{name}_integral"] = curve_in

        # kb is handled separately and non-dimensionalized here: k* = kb / U^2.
        kb_seg_dimless = kb_2d[: head_idx + 1, :] / (self.U ** 2)
        kb_avg, kb_height = self._vertical_average_to_zerocity_zero(
            kb_seg_dimless,
            y_axis,
            ubx_2d[: head_idx + 1, :],
        )
        curves["TKE_avg"] = kb_avg
        curves["TKE_height"] = kb_height
        curves["TKE_integral"] = self._vertical_integral(kb_seg_dimless, y_axis)

        return pd.DataFrame(curves)

    @staticmethod
    def _format_plot_label(column_name: str) -> str:
        """Map output column names to MathText labels for legends.

        Unknown names fall back to raw column strings so new terms still plot.
        """
        label_map = {

        "convection_avg": r"$\left\langle C^* \right\rangle_d$",
        "G_avg": r"$\left\langle G^* \right\rangle_d$",
        "density_gradient_avg": r"$\left\langle \nabla \rho^* \right\rangle_d$",
        "dissipation_avg": r"$\left\langle \varepsilon^* \right\rangle_d$",  # 建议用 \varepsilon 替代 \epsilon
        "diff_avg": r"$\left\langle D^* \right\rangle_d$",
        "drag1_avg": r"$\left\langle F_{d1}^* \right\rangle_d$", # 上下标同时存在时，排版会更紧凑
        "drag2_avg": r"$\left\langle F_{d2}^* \right\rangle_d$",
        "drag3_avg": r"$\left\langle F_{d3}^* \right\rangle_d$",
        "ksource_avg": r"$\left\langle \mathrm{RHS}^* \right\rangle_d$", # 使用 \mathrm 转为正体
        "dkdtof_avg": r"$\left\langle \frac{\partial k^*}{\partial t} \right\rangle_d$", # 替换为标准分数形式
        "ratio_avg": r"$\left\langle \frac{F_d^* + \varepsilon^*}{G^*} \right\rangle_d$", # 替换文字 Drag，改为分数
        "TKE_avg": r"$\left\langle k^* \right\rangle_d$",
        "residual_avg": r"$\left\langle \mathrm{Res}^* \right\rangle_d$" # 简写并转为正体

        
            
        }
        if column_name.endswith("_integral"):
            base = column_name[: -len("_integral")]
            avg_key = f"{base}_avg"
            if avg_key in label_map:
                return f"{label_map[avg_key]} (int)"
            return f"{base} integral"
        return label_map.get(column_name, column_name)

    @staticmethod
    def _legend_if_any(ax, **kwargs):
        """Only draw a legend when at least one labeled line exists."""
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)

    @staticmethod
    def _as_curve_column(name: str) -> str:
        """Convert a variable key to its curve column name, e.g. G -> G_avg.

        This allows users to configure comparison_columns with or without _avg.
        """
        return name if name.endswith("_avg") else f"{name}_avg"

    def _save_outputs(self, time_v: float, df_curve: pd.DataFrame):
        """Save per-time CSV and three default summary figures."""
        os.makedirs(self.output_dir, exist_ok=True)

        time_tag = self._time_tag(time_v)
        time_label = self._time_label(time_v)

        csv_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_tag}.csv")
        df_curve.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        if not self.save_curve_png:
            return

        # Figure 1: averaged curves only.

        png_dir = os.path.join(self.output_dir, f"curve_t{time_tag}")
        os.makedirs(png_dir, exist_ok=True)
        plot_mask = (df_curve["x_dime"].to_numpy(dtype=float) >= 0.0) & (
            df_curve["x_dime"].to_numpy(dtype=float) <= self.x_dime_max
        )
        df_plot = df_curve.loc[plot_mask].copy()

        # Figure 1a: TKE only, saved separately for each time.
        if "TKE_avg" in df_plot.columns:
            fig_k, ax_k = plt.subplots(figsize=self.fig_size)
            ax_k.plot(
                df_plot["x_dime"],
                df_plot["TKE_avg"],
                linewidth=self.curve_lw,
                label=self._format_plot_label("TKE_avg"),
            )
            ax_k.set_title(f"TKE at {time_label}", fontsize=22)
            ax_k.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
            ax_k.set_xlim(self.x_dime_max, 0.0)
            # ax_k.set_ylabel(r"$\langle k^* \rangle_d$", fontsize=20)
            ax_k.tick_params(axis="both", labelsize=18)
            ax_k.grid(True, linestyle="--", alpha=0.35)
            # 1. 设置科学计数法的触发阈值 (让 0.001 也能显示为 1e-3)
            # scilimits=(-2, 3) 表示：数量级小于 10^-2 (即0.01) 或大于 10^3 时触发统一的科学计数法
            ax_k.ticklabel_format(style='sci', axis='y', scilimits=(-1, 3))

            # 2. 获取左上角那个 '1e-X' 文本对象，并强制修改它的字体大小
            offset_text = ax_k.yaxis.get_offset_text()
            offset_text.set_fontsize(16)
            self._legend_if_any(ax_k, fontsize=14, ncol=1, loc="upper left")
            fig_k.tight_layout()

            fig_k_path = os.path.join(png_dir, f"TKE_only_t{time_tag}.png")
            fig_k.savefig(fig_k_path, bbox_inches='tight', dpi=300)
            plt.close(fig_k)
            print(f"Saved Figure: {fig_k_path}")

        fig, ax = plt.subplots(figsize=self.fig_size)
        for col in df_plot.columns:
            if not col.endswith("_avg"):
                continue
            if col in ("TKE_avg", "dkdtof_avg", "ksource_avg", "kresidual_avg", "ratio_avg", "convection_avg"):
                continue
            linestyle = "--" if col in ("drag2_avg", "drag3_avg") else "-"

            ax.plot(
                    df_plot["x_dime"],
                    df_plot[col],
                    linewidth=self.curve_lw,
                    linestyle=linestyle,
                    label=self._format_plot_label(col),
                )

        ax.set_title(f"TKE Budget Terms (Vertical Average) at {time_label}", fontsize=22)
        ax.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
        ax.set_xlim(self.x_dime_max, 0.0)
        # ax.set_ylabel("Vertical average", fontsize=14)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, linestyle="--", alpha=0.35)
        # 1. 设置科学计数法的触发阈值 (让 0.001 也能显示为 1e-3)
        # scilimits=(-2, 3) 表示：数量级小于 10^-2 (即0.01) 或大于 10^3 时触发统一的科学计数法
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-1, 3))

        # 2. 获取左上角那个 '1e-X' 文本对象，并强制修改它的字体大小
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(16)
        self._legend_if_any(ax, fontsize=14, ncol=3, loc="upper left")
        fig.tight_layout()

        fig_path = os.path.join(png_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_tag}.png")
        fig.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

        # Figure 1b: integral curves only.
        fig_int, ax_int = plt.subplots(figsize=self.fig_size)
        for col in df_plot.columns:
            if not col.endswith("_integral"):
                continue
            if col in ("TKE_integral", "dkdtof_integral", "ksource_integral", "kresidual_integral", "ratio_integral", "convection_integral"):
                continue
            linestyle = "--" if col in ("drag2_integral", "drag3_integral") else "-"
            ax_int.plot(
                df_plot["x_dime"],
                df_plot[col],
                linewidth=self.curve_lw,
                label=self._format_plot_label(col),
            )

        ax_int.set_title(f"TKE Budget Terms (Vertical Integral) at {time_label}", fontsize=22)
        ax_int.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
        ax_int.set_xlim(self.x_dime_max, 0.0)
        # ax_int.set_ylabel("Vertical integral", fontsize=14)
        ax_int.tick_params(axis="both", labelsize=18)
        ax_int.grid(True, linestyle="--", alpha=0.35)
        self._legend_if_any(ax_int, fontsize=14, ncol=3, loc="upper left")
        fig_int.tight_layout()

        fig_int_path = os.path.join(png_dir, f"TKE_Budget_SpanAvg_VInt_t{time_tag}.png")
        fig_int.savefig(fig_int_path, bbox_inches='tight', dpi=300)
        plt.close(fig_int)
        print(f"Saved Figure: {fig_int_path}")

        # Figure 1c: selection height curves.
        fig_h, ax_h = plt.subplots(figsize=self.fig_size)
        height_cols = [col for col in df_plot.columns if col.endswith("_height")]
        for col in height_cols:
            ax_h.plot(
                df_plot["x_dime"],
                df_plot[col],
                linewidth=self.curve_lw,
                label=self._format_plot_label(col),
            )

        ax_h.set_title(f"TKE Budget Selection Height at {time_label}", fontsize=22)
        ax_h.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
        ax_h.set_xlim(self.x_dime_max, 0.0)
        ax_h.set_ylabel("Height (m)", fontsize=20)
        ax_h.tick_params(axis="both", labelsize=18)
        ax_h.grid(True, linestyle="--", alpha=0.35)
        fig_h.tight_layout()

        fig_h_path = os.path.join(png_dir, f"TKE_Budget_SpanAvg_Height_t{time_tag}.png")
        fig_h.savefig(fig_h_path, bbox_inches='tight', dpi=300)
        plt.close(fig_h)
        print(f"Saved Figure: {fig_h_path}")

        # Figure 2: residual-only diagnostic.
        fig2, ax2 = plt.subplots(figsize=self.fig_size)
        for col in ("kresidual_avg",):
            if col in df_plot.columns:
                ax2.plot(df_plot["x_dime"], df_plot[col], linewidth=self.curve_lw, label=self._format_plot_label(col))

        ax2.set_title(f"Residual at {time_label}", fontsize=22)
        ax2.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
        ax2.set_xlim(self.x_dime_max, 0.0)
        # ax2.set_ylabel("Vertical average", fontsize=20)
        ax2.tick_params(axis="both", labelsize=18)
        ax2.grid(True, linestyle="--", alpha=0.35)
        self._legend_if_any(ax2, fontsize=14, ncol=2, loc="upper left")
        fig2.tight_layout()

        fig2_path = os.path.join(png_dir, f"Residual_t{time_tag}.png")
        fig2.savefig(fig2_path, bbox_inches='tight', dpi=300)
        plt.close(fig2)
        print(f"Saved Figure: {fig2_path}")

        # Figure 3: selected RHS-related terms.
        fig3, ax3 = plt.subplots(figsize=self.fig_size)
        for col in ("dkdtof_avg", "convection_avg", "ksource_avg"):
            if col in df_plot.columns:
                ax3.plot(df_plot["x_dime"], df_plot[col], linewidth=self.curve_lw, label=self._format_plot_label(col))

        ax3.set_title(f"TKE Budget  at {time_label}", fontsize=22)
        ax3.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
        ax3.set_xlim(self.x_dime_max, 0.0)
        # ax3.set_ylabel("Vertical average", fontsize=20)
        ax3.tick_params(axis="both", labelsize=18)
        ax3.grid(True, linestyle="--", alpha=0.35)
        # 2. 获取左上角那个 '1e-X' 文本对象，并强制修改它的字体大小
        offset_text = ax3.yaxis.get_offset_text()
        offset_text.set_fontsize(16)
        self._legend_if_any(ax3, fontsize=14, ncol=3, loc="upper left")
        fig3.tight_layout()

        fig3_path = os.path.join(png_dir, f"TKE_Budget_t{time_tag}.png")
        fig3.savefig(fig3_path, bbox_inches='tight', dpi=300)
        plt.close(fig3)
        print(f"Saved Figure: {fig3_path}")

        # Figure 4: ratio curve on the averaged 1D data.
        if {"G_avg", "dissipation_avg", "drag1_avg", "ratio_avg"}.issubset(df_curve.columns):
            g_vals = df_plot["G_avg"].to_numpy(dtype=float)
            d_vals = df_plot["dissipation_avg"].to_numpy(dtype=float)
            dr_vals = df_plot["drag1_avg"].to_numpy(dtype=float)
            ratio_numer = dr_vals + d_vals
            ratio = df_plot["ratio_avg"].to_numpy(dtype=float)
            k_vals = df_plot["TKE_avg"].to_numpy(dtype=float)

            ratio_df = pd.DataFrame(
                {
                    "x_dime": df_plot["x_dime"].to_numpy(dtype=float),
                    "G_avg": g_vals,
                    "drag1_plus_dissipation_avg": ratio_numer,
                    "ratio": ratio,
                    "TKE_avg": k_vals,
                }
            )
            ratio_csv_path = os.path.join(png_dir, f"Drag_plus_Dissipation_over_G_t{time_tag}.csv")
            ratio_df.to_csv(ratio_csv_path, index=False)
            print(f"Saved CSV: {ratio_csv_path}")

            fig4, ax4 = plt.subplots(figsize=self.fig_size)
            ax4.plot(
                df_plot["x_dime"],
                ratio,
                linewidth=self.curve_lw,
            )
            ax4.set_title(f"(Drag + Dissipation) / G at {time_label}", fontsize=22)
            ax4.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
            ax4.set_ylabel(r"$\zeta$", fontsize=20)
            ax4.set_xlim(self.x_dime_max, 0.0)
            ax4.tick_params(axis="both", labelsize=18)
            ax4.grid(True, linestyle="--", alpha=0.35)
            fig4.tight_layout()

            fig4_path = os.path.join(png_dir, f"Drag_plus_Dissipation_over_G_t{time_tag}.png")
            fig4.savefig(fig4_path, bbox_inches='tight', dpi=300)
            plt.close(fig4)
            print(f"Saved Figure: {fig4_path}")

    def _save_2d_clouds(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        alpha_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ):
        """Save 2D contour cloud for every loaded term at one time step."""
        time_tag = self._time_tag(time_v)
        time_label = self._time_label(time_v)
        cloud_dir = os.path.join(self.output_dir, f"clouds_t{time_tag}")
        os.makedirs(cloud_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)
        y_vals = y_axis
        alpha_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)[mask, :]

        xx, yy = np.meshgrid(x_dime, y_vals, indexing="ij")

        for name, field in terms_2d.items():
            field_seg = field[: head_idx + 1, :][mask, :]
            field_valid = field_seg[np.isfinite(field_seg)]
            if field_valid.size == 0:
                continue

            p_low, p_high = self.cloud_percentile
            vmin = float(np.percentile(field_valid, p_low))
            vmax = float(np.percentile(field_valid, p_high))
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
                vmin = float(np.nanmin(field_valid))
                vmax = float(np.nanmax(field_valid))
            if vmax <= vmin:
                continue

            levels = np.linspace(vmin, vmax, self.cloud_levels)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if (vmin < 0.0 < vmax) else None

            fig, ax = plt.subplots(figsize=self.cloud_fig_size)
            cf = ax.contourf(xx, yy, field_seg, levels=levels, cmap="coolwarm", norm=norm, extend="both")
            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label(f"{name} (x1000)")

            alpha_valid = alpha_seg[np.isfinite(alpha_seg)]
            if alpha_valid.size > 0:
                a_min = float(np.nanmin(alpha_valid))
                a_max = float(np.nanmax(alpha_valid))
                if a_min <= self.alpha_threshold <= a_max:
                    ax.contour(
                        xx,
                        yy,
                        alpha_seg,
                        levels=[self.alpha_threshold],
                        colors="k",
                        linestyles="--",
                        linewidths=1.0,
                    )

            ax.set_title(f"{name} 2D Cloud at {time_label}", fontsize=22)
            ax.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
            ax.set_ylabel("y (m)", fontsize=20)
            ax.set_xlim(float(np.max(x_dime)), 0.0)
            ax.set_ylim(float(np.min(y_vals)), float(np.max(y_vals)))
            ax.tick_params(axis="both", labelsize=18)

            fig.tight_layout()
            out_path = os.path.join(cloud_dir, f"{name}_2D_t{time_tag}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Saved Figure: {out_path}")

    @staticmethod
    def _write_structured_grid_vtk(
        out_path: str,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        scalar_name: str,
        scalar_field: np.ndarray,
    ):
        """Write one 2D scalar field as legacy VTK STRUCTURED_GRID."""
        nx, ny = scalar_field.shape
        z_2d = np.zeros_like(scalar_field)

        with open(out_path, "w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("TKE budget 2D field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"POINTS {nx * ny} float\n")

            for j in range(ny):
                for i in range(nx):
                    f.write(f"{float(x_2d[i, j]):.9e} {float(y_2d[i, j]):.9e} {float(z_2d[i, j]):.9e}\n")

            f.write(f"POINT_DATA {nx * ny}\n")
            f.write(f"SCALARS {scalar_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    val = float(scalar_field[i, j])
                    if not np.isfinite(val):
                        val = -9999.0
                    f.write(f"{val:.9e}\n")

    def _save_2d_vtk(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        alpha_2d: np.ndarray,
        vort_x_2d: np.ndarray,
        vort_y_2d: np.ndarray,
        vort_z_2d: np.ndarray,
        head_idx: int,
        head_x: float,
        Rig_2d: np.ndarray,
        gradUb_ux_dz_2d: np.ndarray,
        gradMixrho_z_2d: np.ndarray,
        lambda2_2d: np.ndarray,
        ub_2d: np.ndarray,
        kb_2d: np.ndarray,
    ):
        """Save every term as a standalone 2D VTK file for Paraview."""
        time_tag = self._time_tag(time_v)
        vtk_dir = os.path.join(self.output_dir, f"vtk_t{time_tag}")
        os.makedirs(vtk_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)
        print(f'VTK y-axis range: {y_axis.min()} to {y_axis.max()} ')
        y_vals = y_axis/0.3
        xx, yy = np.meshgrid(x_dime, y_vals, indexing="ij")

        alpha_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)[mask, :]
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"alpha_a_t{time_tag}.vtk"),
            xx,
            yy,
            "alpha_a",
            alpha_seg,
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"alpha_minus_threshold_t{time_tag}.vtk"),
            xx,
            yy,
            "alpha_minus_threshold",
            alpha_seg - self.alpha_threshold,
        )

        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"vorticity_Ub_x_t{time_tag}.vtk"),
            xx,
            yy,
            "vorticity_Ub_x",   
            vort_x_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"vorticity_Ub_y_t{time_tag}.vtk"),
            xx,         
            yy,
            "vorticity_Ub_y",
            vort_y_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"vorticity_Ub_z_t{time_tag}.vtk"),
            xx,
            yy,
            "vorticity_Ub_z",
            vort_z_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"Rig_t{time_tag}.vtk"),
            xx,
            yy,
            "Rig",
            Rig_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"gradUb_ux_dz_t{time_tag}.vtk"),
            xx,
            yy,
            "gradUb_ux_dz",
            gradUb_ux_dz_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"gradMixrho_z_t{time_tag}.vtk"),
            xx,
            yy,
            "gradMixrho_z",
            gradMixrho_z_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"lambda2_t{time_tag}.vtk"),
            xx,
            yy,
            "lambda2",
            lambda2_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"ubx_t{time_tag}.vtk"),
            xx,
            yy,
            "ub",
            ub_2d[: head_idx + 1, :][mask, :],
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"kb_t{time_tag}.vtk"),
            xx,
            yy,
            "kb",
            kb_2d[: head_idx + 1, :][mask, :],
        )


        for name, field in terms_2d.items():
            field_seg = field[: head_idx + 1, :][mask, :]
            out_path = os.path.join(vtk_dir, f"{name}_t{time_tag}.vtk")
            self._write_structured_grid_vtk(out_path, xx, yy, name, field_seg)

        if "G" in terms_2d and "dissipation" in terms_2d:
            g_seg = terms_2d["G"][: head_idx + 1, :][mask, :]
            d_seg = terms_2d["dissipation"][: head_idx + 1, :][mask, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.divide(g_seg, d_seg, out=np.full_like(g_seg, np.nan), where=np.abs(d_seg) > 1e-20)

            signed_log = np.full_like(ratio, np.nan)
            valid_mask = np.isfinite(ratio)
            signed_log[valid_mask] = np.log1p(np.abs(ratio[valid_mask]))

            self._write_structured_grid_vtk(
                os.path.join(vtk_dir, f"signed_log_G_over_dissipation_t{time_tag}.vtk"),
                xx,
                yy,
                "signed_log_G_over_dissipation",
                signed_log,
            )

        print(f"Saved VTK directory: {vtk_dir}")

    def _save_comparison(self, comparison_frames):
        """Save separate figures for selected variables across all time steps.

        Each output figure contains one variable and multiple time curves.
        """
        if not comparison_frames:
            print("No valid data found. Skip multi-variable comparison figure.")
            return

        if not self.save_comparison_png:
            return

        # Normalize user requests to DataFrame column names.
        requested_columns = [self._as_curve_column(col) for col in self.comparison_columns]
        if self.num_comparison_variables is not None:
            # Limit to first N requests if user configured a cap.
            requested_columns = requested_columns[: max(0, int(self.num_comparison_variables))]

        available_columns = [
            col for col in requested_columns if any(col in df.columns for _, df in comparison_frames)
        ]
        if not available_columns:
            print("None of requested comparison columns exist in processed outputs. Skip comparison figure.")
            return

        for col in available_columns:
            fig, ax = plt.subplots(figsize=self.fig_size)
            for time_v, df_curve in comparison_frames:
                if col not in df_curve.columns:
                    continue
                xvals = df_curve["x_dime"].to_numpy(dtype=float)
                yvals = df_curve[col].to_numpy(dtype=float)
                mask = (xvals >= 0.0) & (xvals <= self.x_dime_max)
                if not np.any(mask):
                    continue
                xvals = xvals[mask]
                yvals = yvals[mask]
                ax.plot(xvals, yvals, linewidth=self.curve_lw, label=self._time_label(time_v))

            ax.set_title(f"{self._format_plot_label(col)} Comparison Across Nondimensional Time", fontsize=22)
            ax.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
            ax.set_xlim(self.x_dime_max, 0.0)
            # ax.set_ylabel("Vertical average", fontsize=20)
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(True, linestyle="--", alpha=0.35)
            self._legend_if_any(ax, fontsize=18, ncol=1, loc="upper left")
            fig.tight_layout()

            fig_path = os.path.join(self.output_dir, f"TKE_Comparison_{col}.png")
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            print(f"Saved Figure: {fig_path}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[pd.DataFrame]:
        """Process one time step and return the depth-averaged curve table."""
        time_label = self._time_label(time_v)
        data_3d = self._load_terms_3d(grid, float(time_v))
        if data_3d is None:
            return None

        # Use spanwise-averaged alpha_a only for front/head detection.
        alpha_a_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            print(f"No alpha.a > threshold ({self.alpha_threshold}) at {time_label}. Skip output.")
            return None

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx}) at {time_label}")

        beta_3d = 1.0 - data_3d.alpha_a

        # Guard against zero/invalid shear to avoid RuntimeWarning and bad ratios.
        shear2 = np.square(data_3d.gradUb_ux_dz)
        numer = -9.81 * data_3d.gradMixrho_z
        valid = np.isfinite(numer) & np.isfinite(shear2) & (shear2 > 1e-20)
        Rig = np.zeros_like(numer, dtype=float)
        Rig = np.divide(numer, shear2*1000, out=np.full_like(numer, np.nan), where=valid)


        Rig_2d = np.mean(Rig, axis=2)
        
        terms_3d = {name: field * beta_3d for name, field in data_3d.terms.items()}
        ub_2d = np.mean(data_3d.Ub, axis=2)
        ubx_2d = np.mean(data_3d.Ub[0], axis=2)
        graduxuz = np.mean(data_3d.gradUb_ux_dz, axis=2)
        gradMixrho_z_2d = np.mean(data_3d.gradMixrho_z, axis=2)
        lambda2_2d = np.mean(data_3d.lambda2, axis=2)
        vorticity_x_2d = np.mean(data_3d.ubvorticity_x, axis=2)
        vorticity_y_2d = np.mean(data_3d.ubvorticity_y, axis=2)
        vorticity_z_2d = np.mean(data_3d.ubvorticity_z, axis=2)
        kb_2d = np.mean(data_3d.kb, axis=2)
        kb_2d_dimless = kb_2d / (self.U**2)

        terms_2d = {
            name: field /(self.U**3/ self.H) 
            for name, field in self._spanwise_average_terms(terms_3d).items()}

        if self.save_vtk:
            self._save_2d_vtk(
                float(time_v),
                data_3d.x_axis,
                data_3d.y_axis,
                terms_2d,
                alpha_a_2d,
                vorticity_x_2d,
                vorticity_y_2d,
                vorticity_z_2d,
                head_idx,
                head_x,
                Rig_2d,
                graduxuz,
                gradMixrho_z_2d,
                lambda2_2d,
                ubx_2d,
                kb_2d_dimless,
            )
        df_curve = self._average_to_curves(data_3d.x_axis, data_3d.y_axis, terms_2d, ubx_2d, head_idx, kb_2d)
        if {"G_avg", "dissipation_avg", "drag1_avg"}.issubset(df_curve.columns):
            numer = df_curve["drag1_avg"].to_numpy(dtype=float) + df_curve["dissipation_avg"].to_numpy(dtype=float)
            denom = df_curve["G_avg"].to_numpy(dtype=float)
            ratio = np.divide(
                numer,
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=np.abs(denom) > 1e-20,
            )
            df_curve["ratio_avg"] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        self._save_outputs(float(time_v), df_curve)
        return df_curve

    def run_analysis(self):
        """Main entry: read mesh once, process all times, then save cross-time comparisons."""
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        comparison_frames = []
        for t in self.times:
            df_curve = self.process_time_step(grid, t)
            if df_curve is None:
                continue
            comparison_frames.append((float(t), df_curve))

        self._save_comparison(comparison_frames)


if __name__ == "__main__":
    analyzer = TKEBudgetAnalyzer()
    analyzer.run_analysis()
