import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d
## 对于TKE的项，也就是load term中的几个，原of输出的结果是/betarhob的 
##在这里先乘回去beta，考虑无量纲结果最后还是会除以rhob，所以等价于计算垂直平均和积分时就都是物理量纲的结果了，最后再统一进行无量纲化处理。

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
    grad_alpha: Optional[np.ndarray] = None
    lambda2: Optional[np.ndarray] = None
    kdivub: Optional[np.ndarray] = None
    drag1_split: Optional[Dict[str, np.ndarray]] = None
    drag1_con: Optional[Dict[str, np.ndarray]] = None
    velocity_diff: Optional[Dict[str, np.ndarray]] = None
    gradMixrho_z: Optional[np.ndarray] = None
    coeff: Optional[np.ndarray] = None


class TKEBudgetAnalyzer:
    def __init__(self):
        # OpenFOAM case directory and output directory for generated CSV/figures.
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090428_1"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4test"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2"
        # self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d09_0604_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        # self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0428_41e3"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_2"
        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0604_2"
        # Physical times to be processed.
        self.times = [5,12,25]

        # Threshold to detect current head position from alpha.a.
        self.alpha_threshold = 1e-3
        # Default style settings shared by all figures.
        self.fig_size = (10, 3.4)
        self.curve_lw = 2.0
        self.x_dime_max = 8
        self.cloud_fig_size = (9, 3.2)
        self.cloud_levels = 121
        self.cloud_percentile = (1.0, 99.0)
        self.alpha_cloud_max = 0.01
        self.alpha_cloud_contour_level = 1e-5
        # If True, VTK scalar values are clipped using the same percentile range as cloud plots.
        self.vtk_match_cloud_percentile = True
        self.save_curve_png = True
        self.save_comparison_png = True
        self.save_vtk = True
        # Variables compared across all selected times in one summary figure set.
        self.comparison_columns = ["G", "convection", "diff", "drag1", "dissipation", "ratio"]
        self.num_comparison_variables = None

        self.U = 0.26
        self.H = 0.3
        # Set to a number to use a uniform gamma/K when OpenFOAM K is constant
        # or not written as a volScalarField. Leave as None to require reading K.
        self.gamma_constant = 9e7
        self.min_vertical_integration_height_ratio = 0.05
        self.debug_upper_limit_time = 25.0
        self.debug_upper_limit_x = 1.804
        self.title_fontsize = 24
        self.legend_fontsize = 22
        self.offset_fontsize = 22
        self.label_fontsize = 24
        self.tick_fontsize = 22
        self.cbar_labelsize = 24
        self.cbar_ticksize = 22
        
        # Smooth only the TKE selection-height curve; set 0 to disable.
        self.tke_height_smooth_sigma = 2.0

        # Mapping: output column name -> OpenFOAM field name.
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
        return f"{float(time_v):g}"

    @staticmethod
    def _nondim_time(time_v: float) -> float:
        return float(time_v) * 0.85

    def _time_tag(self, time_v: float) -> str:
        return f"{time_v:.2f}"

    def _time_label(self, time_v: float) -> str:
        return rf"$t*={self._nondim_time(time_v):.2f}$"

    @staticmethod
    def _build_grid_cache(X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray) -> Dict[str, np.ndarray]:
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
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    def _read_scalar_or_constant(
        self,
        time_dir: str,
        field_name: str,
        fallback_value: Optional[float],
        sort_idx: np.ndarray,
        nx: int,
        ny: int,
        nz: int,
    ) -> np.ndarray:
        try:
            raw = np.asarray(fluidfoam.readscalar(self.sol, time_dir, field_name), dtype=float)
            if raw.size == 1:
                return np.full((nx, ny, nz), float(raw.reshape(-1)[0]), dtype=float)
            return self._reshape_sorted(raw, sort_idx, nx, ny, nz)
        except Exception as exc:
            if fallback_value is None:
                raise
            print(f"Use constant {field_name}={fallback_value} at {time_dir}; read failed: {exc}")
            return np.full((nx, ny, nz), float(fallback_value), dtype=float)

    @staticmethod
    def _compute_lambda2(grad_u: np.ndarray) -> np.ndarray:
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
        if hasattr(np, "trapezoid"):
            return np.trapezoid(field2d, x=y_coords, axis=1)
        else:
            return np.trapz(field2d, x=y_coords, axis=1)

    @staticmethod
    def _vertical_average_to_zerocity_zero(
        field2d: np.ndarray,
        y_coords: np.ndarray,
        ubx2d: np.ndarray,
        min_height: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        nx = field2d.shape[0]
        y_lower = 0.0005
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
            u_sel = u_valid[active_mask]

            if y_sel.size < 2:
                heights[i] = 0.0
                out[i] = 0.0
                continue

            if hasattr(np, "trapezoid"):
                numerator = float(np.trapezoid(f_sel, x=y_sel))
                int_u = float(np.trapezoid(u_sel, x=y_sel))
                int_u2 = float(np.trapezoid(u_sel**2, x=y_sel))
            else:
                numerator = float(np.trapz(f_sel, x=y_sel))
                int_u = float(np.trapz(u_sel, x=y_sel))
                int_u2 = float(np.trapz(u_sel**2, x=y_sel))

            velocity_height = int_u**2 / int_u2 if abs(int_u2) > 1e-20 else 0.0
            heights[i] = velocity_height
            if velocity_height > max(min_height, 1e-12):
                out[i] = numerator / velocity_height
            else:
                out[i] = 0.0

        return out, heights

    def _smooth_1d_nanaware(self, values: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.0:
            return values.copy()

        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            return values.copy()

        weights = gaussian_filter1d(valid.astype(float), sigma=sigma, mode="nearest")
        smoothed = gaussian_filter1d(
            np.where(valid, values, 0.0),
            sigma=sigma,
            mode="nearest",
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = smoothed / weights
        smoothed[weights < 1e-8] = np.nan
        return smoothed

    def _trim_x_dime(self, x_seg: np.ndarray, x_head: float):
        x_dime = (x_head - x_seg) / self.H
        mask = (x_dime >= 0.0) & (x_dime <= self.x_dime_max)
        return x_seg[mask], x_dime[mask], mask

    def _print_upper_limit_debug(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        ubx_2d: np.ndarray,
    ) -> None:
        if self.debug_upper_limit_time is None or self.debug_upper_limit_x is None:
            return
        if not np.isclose(float(time_v), float(self.debug_upper_limit_time)):
            return

        i = int(np.argmin(np.abs(x_axis - self.debug_upper_limit_x)))
        u_profile = ubx_2d[i]
        valid = np.isfinite(u_profile) & np.isfinite(y_axis)
        y_valid = y_axis[valid]
        u_valid = u_profile[valid]
        y_lower = 0.001

        zero_j = None
        zero_y = None
        for j in range(len(u_valid) - 1):
            if y_valid[j] < y_lower:
                continue
            if u_valid[j] > 0.0 and u_valid[j + 1] <= 0.0:
                zero_j = j + 1
                zero_y = float(y_valid[j + 1])
                break

        y_upper = float(y_valid[-1]) if zero_y is None else float(zero_y)
        active_mask = (y_valid >= y_lower) & (y_valid <= y_upper)
        y_sel = y_valid[active_mask]
        u_sel = u_valid[active_mask]

        velocity_height = np.nan
        int_u = np.nan
        int_u2 = np.nan
        if y_sel.size >= 2:
            if hasattr(np, "trapezoid"):
                int_u = float(np.trapezoid(u_sel, x=y_sel))
                int_u2 = float(np.trapezoid(u_sel**2, x=y_sel))
            else:
                int_u = float(np.trapz(u_sel, x=y_sel))
                int_u2 = float(np.trapz(u_sel**2, x=y_sel))
            velocity_height = int_u**2 / int_u2 if abs(int_u2) > 1e-20 else 0.0

        print(
            "[upper-limit-debug] "
            f"t={time_v:g}, target_x={self.debug_upper_limit_x:.6g}, "
            f"nearest_x={float(x_axis[i]):.9g}, x_index={i}, "
            f"zero_index={zero_j}, y_upper={y_upper:.9g}, "
            f"int_u={int_u:.9g}, int_u2={int_u2:.9g}, "
            f"velocity_height={velocity_height:.9g}, height/H={velocity_height / self.H:.9g}"
        )
        if zero_j is not None and zero_j > 0:
            print(
                "[upper-limit-debug] "
                f"crossing bracket: y_before={float(y_valid[zero_j - 1]):.9g}, "
                f"u_before={float(u_valid[zero_j - 1]):.9g}, "
                f"y_after={float(y_valid[zero_j]):.9g}, "
                f"u_after={float(u_valid[zero_j]):.9g}"
            )

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if len(valid_x) == 0:
            return None
        return int(valid_x.max())

    def _load_terms_3d(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepTerms3D]:
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
        
        gradUb_ux_dz = gradUb[3]
        gradMixrho_z = gradMixrho[1]
        kdivub = (gradUb[0] + gradUb[4] + gradUb[8])*kb
        lambda2 = self._compute_lambda2(gradUb)
        ubvorticity = self._reshape_sorted(ubvorticity, sort_idx, nx, ny, nz)
        ubvorticity_x = ubvorticity[0]
        ubvorticity_y = ubvorticity[1]
        ubvorticity_z = ubvorticity[2]

        drag1_split = None
        drag1_con = None
        velocity_diff_split = None
        grad_alpha = None
        coeff = None
        try:
            ua_raw = fluidfoam.readvector(self.sol, time_dir, "U.a")
            grad_alpha_raw = fluidfoam.readvector(self.sol, time_dir, "grad(alpha.a)")
            nut_raw = fluidfoam.readscalar(self.sol, time_dir, "nut.b")
            sus_raw = fluidfoam.readscalar(self.sol, time_dir, "SUS")
            alpha_b_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.b")

            Ua = self._reshape_sorted(ua_raw, sort_idx, nx, ny, nz)
            grad_alpha = self._reshape_sorted(grad_alpha_raw, sort_idx, nx, ny, nz)
            nut = self._reshape_sorted(nut_raw, sort_idx, nx, ny, nz)
            sus = self._reshape_sorted(sus_raw, sort_idx, nx, ny, nz)
            gamma = self._read_scalar_or_constant(
                time_dir,
                "K",
                self.gamma_constant,
                sort_idx,
                nx,
                ny,
                nz,
            ) / 1e3
            alpha_b = self._reshape_sorted(alpha_b_raw, sort_idx, nx, ny, nz)
          

            with np.errstate(divide="ignore", invalid="ignore"):
                coeff = gamma * np.divide(nut, sus, out=np.full_like(nut, np.nan), where=np.abs(sus) > 1e-12)
                coeff = np.divide(coeff, alpha_b, out=np.full_like(coeff, np.nan), where=np.abs(alpha_b) > 1e-12)

            velocity_diff = Ub - Ua
            velocity_diff_split = {
                "velocity_diff_x": np.nan_to_num(velocity_diff[0], nan=0.0, posinf=0.0, neginf=0.0),
                "velocity_diff_y": np.nan_to_num(velocity_diff[1], nan=0.0, posinf=0.0, neginf=0.0),
            }
            drag1_split = {
                "drag1_split_x": np.nan_to_num(velocity_diff[0] * grad_alpha[0], nan=0.0, posinf=0.0, neginf=0.0),
                "drag1_split_y": np.nan_to_num(velocity_diff[1] * grad_alpha[1], nan=0.0, posinf=0.0, neginf=0.0),
            }
            drag1_con = {
                "drag1_con_x": np.nan_to_num(coeff * velocity_diff[0] * grad_alpha[0], nan=0.0, posinf=0.0, neginf=0.0),
                "drag1_con_y": np.nan_to_num(coeff * velocity_diff[1] * grad_alpha[1], nan=0.0, posinf=0.0, neginf=0.0),
            
            }
        except Exception as exc:
            print(f"Skip drag1 split fields at {time_tag}: {exc}")

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
            grad_alpha=grad_alpha,
            lambda2=lambda2,
            kb=kb,
            kdivub=kdivub,
            drag1_split=drag1_split,
            drag1_con=drag1_con,
            velocity_diff=velocity_diff_split,
            gradMixrho_z=gradMixrho_z,
            coeff=coeff
        )

    @staticmethod
    def _spanwise_average_terms(terms_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: np.mean(field, axis=2) for name, field in terms_3d.items()}

    def _average_to_curves(
        self,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        ubx_2d: np.ndarray,
        head_idx: int,
        kb_2d: np.ndarray,
        kdiv_2d: np.ndarray,
    ) -> pd.DataFrame:
        x_seg = x_axis[: head_idx + 1]
        x_head = x_axis[head_idx]
        x_dime = (x_head - x_seg) / self.H

        curves = {
            "x": x_seg,
            "x_dime": x_dime,
        }
        min_height = self.min_vertical_integration_height_ratio * self.H

        # 此时传入的 terms_2d 和 kb_2d 均为纯物理原有量纲数据
        for name, field in terms_2d.items():
            field_seg = field[: head_idx + 1, :]

            # 1) 物理量纲下的垂直平均
            curve_avg, curve_height = self._vertical_average_to_zerocity_zero(
                field_seg, y_axis, ubx_2d[: head_idx + 1, :], min_height=min_height
            )
            curves[f"{name}_avg"] = curve_avg

            # 2) 物理量纲下的全深垂直积分 (单位: 原始单位 * m)
            # curve_in = self._vertical_integral(field_seg, y_axis)
            # curves[f"{name}_integral"] = curve_in

        # 处理 TKE (kb)
        kb_seg = kb_2d[: head_idx + 1, :]
        kb_avg, kb_height = self._vertical_average_to_zerocity_zero(
            kb_seg,
            y_axis,
            ubx_2d[: head_idx + 1, :],
            min_height=min_height,
        )
        curves["TKE_avg"] = kb_avg
        curves["height_raw"] = kb_height
        curves["height"] = self._smooth_1d_nanaware(
            kb_height,
            self.tke_height_smooth_sigma,
        )
        # curves["TKE_integral"] = self._vertical_integral(kb_seg, y_axis)

        # 处理 kdivub
        kdiv_seg = kdiv_2d[: head_idx + 1, :]
        kdiv_avg, _ = self._vertical_average_to_zerocity_zero(
            kdiv_seg,
            y_axis,
            ubx_2d[: head_idx + 1, :],
            min_height=min_height,
        )
        curves["kdiv_avg"] = kdiv_avg

        return pd.DataFrame(curves)

    def _nondimensionalize_output_columns(self, df_curve: pd.DataFrame) -> pd.DataFrame:
        """Apply final nondimensionalization to output columns at the very end."""
        df_out = df_curve.copy()
        
        # 1. 尺度/高度项无量纲化 (除以 H)
        for col in ("height", "height_raw"):
            if col in df_out.columns:
                df_out[col] = df_out[col].to_numpy(dtype=float) / self.H
                
        # 2. TKE 项无量纲化 (TKE_avg 属于速度平方项, 除以 U^2; 积分项需额外除以 H)
        if "TKE_avg" in df_out.columns:
            df_out["TKE_avg"] = df_out["TKE_avg"].to_numpy(dtype=float) / (self.U**2)
        if "TKE_integral" in df_out.columns:
            df_out["TKE_integral"] = df_out["TKE_integral"].to_numpy(dtype=float) / (self.U**2 * self.H)
            
        # 3. 能量收支平衡各项(Budget Terms)无量纲化
        # 平均项单位为 W/kg (m^2/s^3), 无量纲基准为 U^3 / H
        # 积分项单位为 m^3/s^3, 对应的无量纲基准为 U^3
        for col in df_out.columns:
            if col.endswith("_avg") and col != "TKE_avg" and col != "ratio_avg":
                df_out[col] = df_out[col].to_numpy(dtype=float) / (self.U**3 / self.H)
            elif col.endswith("_integral") and col != "TKE_integral":
                df_out[col] = df_out[col].to_numpy(dtype=float) / (self.U**3)
                
        return df_out

    @staticmethod
    def _format_plot_label(column_name: str) -> str:
        label_map = {
            "convection_avg": r"$\left\langle C^* \right\rangle_d$",
            "G_avg": r"$\left\langle G^* \right\rangle_d$",
            "density_gradient_avg": r"$\left\langle \nabla \rho^* \right\rangle_d$",
            "dissipation_avg": r"$\left\langle \varepsilon^* \right\rangle_d$",
            "diff_avg": r"$\left\langle D^* \right\rangle_d$",
            "drag1_avg": r"$\left\langle F_{d1}^* \right\rangle_d$",
            "drag2_avg": r"$\left\langle F_{d2}^* \right\rangle_d$",
            "drag3_avg": r"$\left\langle F_{d3}^* \right\rangle_d$",
            "ksource_avg": r"$\left\langle \mathrm{RHS}^* \right\rangle_d$",
            "dkdtof_avg": r"$\left\langle \partial k^*/ \partial t \right\rangle_d$",
            "ratio_avg": r"$\left\langle \frac{F_d^* + \varepsilon^*}{G^*} \right\rangle_d$",
            "TKE_avg": r"$\left\langle k^* \right\rangle_d$",
            "height": r"$h/H$ (smooth)",
            "height_raw": r"$h/H$ (raw)",
            "kresidual_avg": r"$\left\langle \mathrm{Res}^* \right\rangle_d$",
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
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)

    @staticmethod
    def _as_curve_column(name: str) -> str:
        return name if name.endswith("_avg") else f"{name}_avg"

    def _save_outputs(self, time_v: float, df_curve: pd.DataFrame):
        os.makedirs(self.output_dir, exist_ok=True)

        time_tag = self._time_tag(time_v)
        time_label = self._time_label(time_v)

        csv_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_tag}.csv")
        df_curve.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        if not self.save_curve_png:
            return

        png_dir = os.path.join(self.output_dir, f"curve_t{time_tag}")
        os.makedirs(png_dir, exist_ok=True)
        plot_mask = (df_curve["x_dime"].to_numpy(dtype=float) >= 0.0) & (
            df_curve["x_dime"].to_numpy(dtype=float) <= self.x_dime_max
        )
        df_plot = df_curve.loc[plot_mask].copy()

        if "TKE_avg" in df_plot.columns:
            fig_k, ax_k = plt.subplots(figsize=self.fig_size)
            ax_k.plot(
                df_plot["x_dime"],
                df_plot["TKE_avg"],
                linewidth=self.curve_lw,
                label=self._format_plot_label("TKE_avg"),
            )
            ax_k.set_title(f"TKE at {time_label}", fontsize=self.title_fontsize)
            ax_k.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
            ax_k.set_xlim(self.x_dime_max, 0.0)
            ax_k.tick_params(axis="both", labelsize=self.tick_fontsize)
            ax_k.grid(True, linestyle="--", alpha=0.35)
            ax_k.ticklabel_format(style="sci", axis="y", scilimits=(-1, 3))

            offset_text = ax_k.yaxis.get_offset_text()
            offset_text.set_fontsize(self.offset_fontsize)
            self._legend_if_any(ax_k, fontsize=self.legend_fontsize, ncol=1, loc="upper left")
            fig_k.tight_layout()

            fig_k_path = os.path.join(png_dir, f"TKE_only_t{time_tag}.png")
            fig_k.savefig(fig_k_path, bbox_inches="tight", dpi=300)
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

        ax.set_title(f"TKE Budget Terms (Vertical Average) at {time_label}", fontsize=self.title_fontsize)
        ax.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
        ax.set_xlim(self.x_dime_max, 0.0)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-1, 3))

        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(self.offset_fontsize)
        self._legend_if_any(ax, fontsize=self.legend_fontsize, ncol=4, loc="upper left")
        fig.tight_layout()

        fig_path = os.path.join(png_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_tag}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

        fig_int, ax_int = plt.subplots(figsize=self.fig_size)
        for col in df_plot.columns:
            if not col.endswith("_integral"):
                continue
            if col in (
                "TKE_integral",
                "dkdtof_integral",
                "ksource_integral",
                "kresidual_integral",
                "ratio_integral",
                "convection_integral",
            ):
                continue
            ax_int.plot(
                df_plot["x_dime"],
                df_plot[col],
                linewidth=self.curve_lw,
                label=self._format_plot_label(col),
            )

        ax_int.set_title(f"TKE Budget Terms (Vertical Integral) at {time_label}", fontsize=self.title_fontsize)
        ax_int.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
        ax_int.set_xlim(self.x_dime_max, 0.0)
        ax_int.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax_int.grid(True, linestyle="--", alpha=0.35)
        self._legend_if_any(ax_int, fontsize=self.legend_fontsize, ncol=3, loc="upper left")
        fig_int.tight_layout()

        fig_int_path = os.path.join(png_dir, f"TKE_Budget_SpanAvg_VInt_t{time_tag}.png")
        fig_int.savefig(fig_int_path, bbox_inches="tight", dpi=300)
        plt.close(fig_int)
        print(f"Saved Figure: {fig_int_path}")

        fig_h, ax_h = plt.subplots(figsize=self.fig_size)
        height_cols = [col for col in ("height", "height_raw") if col in df_plot.columns]
        for col in height_cols:
            linestyle = "--" if col == "height_raw" else "-"
            ax_h.plot(
                df_plot["x_dime"],
                df_plot[col],
                linewidth=self.curve_lw,
                linestyle=linestyle,
                label=self._format_plot_label(col),
            )

        ax_h.set_xlabel(r"$(x_f-x)/H_0$", fontsize=self.label_fontsize)
        ax_h.set_xlim(self.x_dime_max, 0.0)
        ax_h.set_ylabel(r"$z^*$", fontsize=self.label_fontsize, rotation=0, labelpad=20)
        ax_h.set_ylim(0.0, 1.0)
        ax_h.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax_h.grid(True, linestyle="--", alpha=0.35)
        fig_h.tight_layout()

        fig_h_path = os.path.join(png_dir, f"TKE_Budget_SpanAvg_Height_t{time_tag}.png")
        fig_h.savefig(fig_h_path, bbox_inches="tight", dpi=300)
        plt.close(fig_h)
        print(f"Saved Figure: {fig_h_path}")

        fig2, ax2 = plt.subplots(figsize=self.fig_size)
        for col in ("kresidual_avg",):
            if col in df_plot.columns:
                ax2.plot(df_plot["x_dime"], df_plot[col], linewidth=self.curve_lw, label=self._format_plot_label(col))

        ax2.set_title(f"Residual at {time_label}", fontsize=self.title_fontsize)
        ax2.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
        ax2.set_xlim(self.x_dime_max, 0.0)
        ax2.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(-1, 3))

        offset_text = ax2.yaxis.get_offset_text()
        offset_text.set_fontsize(self.offset_fontsize)

        self._legend_if_any(ax2, fontsize=self.legend_fontsize, ncol=2, loc="upper left")
        fig2.tight_layout()

        fig2_path = os.path.join(png_dir, f"Residual_t{time_tag}.png")
        fig2.savefig(fig2_path, bbox_inches="tight", dpi=300)
        plt.close(fig2)
        print(f"Saved Figure: {fig2_path}")

        fig3, ax3 = plt.subplots(figsize=self.fig_size)
        for col in ("dkdtof_avg", "convection_avg", "ksource_avg"):
            if col in df_plot.columns:
                ax3.plot(df_plot["x_dime"], df_plot[col], linewidth=self.curve_lw, label=self._format_plot_label(col))

        ax3.set_title(f"TKE Budget   at {time_label}", fontsize=self.title_fontsize)
        ax3.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
        ax3.set_xlim(self.x_dime_max, 0.0)
        ax3.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax3.grid(True, linestyle="--", alpha=0.35)
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(-1, 3))
        offset_text = ax3.yaxis.get_offset_text()
        offset_text.set_fontsize(self.offset_fontsize)
        self._legend_if_any(ax3, fontsize=self.legend_fontsize, ncol=3, loc="upper left")
        fig3.tight_layout()

        fig3_path = os.path.join(png_dir, f"TKE_Budget_t{time_tag}.png")
        fig3.savefig(fig3_path, bbox_inches="tight", dpi=300)
        plt.close(fig3)
        print(f"Saved Figure: {fig3_path}")

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
            ax4.set_title(f"(Drag + Dissipation) / G at {time_label}", fontsize=self.title_fontsize)
            ax4.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
            ax4.set_ylabel(r"$\zeta$", fontsize=self.label_fontsize)
            ax4.set_xlim(self.x_dime_max, 0.0)
            ax4.tick_params(axis="both", labelsize=self.tick_fontsize)
            ax4.grid(True, linestyle="--", alpha=0.35)
            fig4.tight_layout()

            fig4_path = os.path.join(png_dir, f"Drag_plus_Dissipation_over_G_t{time_tag}.png")
            fig4.savefig(fig4_path, bbox_inches="tight", dpi=300)
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

            ax.set_title(f"{name} 2D Cloud at {time_label}", fontsize=self.title_fontsize)
            ax.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
            ax.set_ylabel("y (m)", fontsize=self.label_fontsize)
            ax.set_xlim(float(np.max(x_dime)), 0.0)
            ax.set_ylim(float(np.min(y_vals)), float(np.max(y_vals)))
            ax.tick_params(axis="both", labelsize=self.tick_fontsize)

            fig.tight_layout()
            out_path = os.path.join(cloud_dir, f"{name}_2D_t{time_tag}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Saved Figure: {out_path}")

    def _save_alpha_height_cloud(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        alpha_2d: np.ndarray,
        head_idx: int,
        head_x: float,
        df_curve: pd.DataFrame,
    ):
        """Save alpha cloud with alpha threshold contour and raw height curve."""
        if "height_raw" not in df_curve.columns:
            return

        time_tag = self._time_tag(time_v)
        cloud_dir = os.path.join(self.output_dir, f"clouds_t{time_tag}")
        os.makedirs(cloud_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        _, x_dime, mask = self._trim_x_dime(x_seg, head_x)
        y_dime = y_axis / self.H
        alpha_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)[mask, :]
        alpha_plot = np.clip(alpha_seg, 0.0, self.alpha_cloud_max)
        xx, yy = np.meshgrid(x_dime, y_dime, indexing="ij")

        fig, ax = plt.subplots(figsize=self.cloud_fig_size)
        cf = ax.contourf(
            xx,
            yy,
            alpha_plot,
            levels=np.linspace(0.0, self.alpha_cloud_max, self.cloud_levels),
            cmap="gray_r",
            extend="neither",
        )
        # cbar = fig.colorbar(cf, ax=ax, pad=0.02)
        # cbar.set_label(r"$\alpha_a$", fontsize=18)
        # cbar.ax.tick_params(labelsize=16)

        alpha_valid = alpha_seg[np.isfinite(alpha_seg)]
        if alpha_valid.size > 0:
            a_min = float(np.nanmin(alpha_valid))
            a_max = float(np.nanmax(alpha_valid))
            if a_min <= self.alpha_cloud_contour_level <= a_max:
                ax.contour(
                    xx,
                    yy,
                    alpha_seg,
                    levels=[self.alpha_cloud_contour_level],
                    colors="0.25",
                    linestyles="--",
                    linewidths=1.2,
                    label = r"$\alpha_s = 10^{-5}$",
                )

        height_mask = (
            (df_curve["x_dime"].to_numpy(dtype=float) >= 0.0)
            & (df_curve["x_dime"].to_numpy(dtype=float) <= self.x_dime_max)
        )
        height_df = df_curve.loc[height_mask, ["x_dime", "height_raw"]].copy()
        height_df = height_df[np.isfinite(height_df["height_raw"].to_numpy(dtype=float))]
        if not height_df.empty:
            ax.plot(
                height_df["x_dime"],
                height_df["height_raw"],
                color="#FF00FF",
                linewidth=1.2,
                label=r"$Z_c^*$",
            )

        ax.set_xlabel(r"$(x_f-x)/H_0$", fontsize=20)
        ax.set_ylabel(r"$z^*$", fontsize=20, rotation=0, labelpad=18)
        ax.set_xlim(max(x_dime), 0.0)
        ax.set_ylim(0.0, 0.5)
        ax.tick_params(axis="both", labelsize=18)
        self._legend_if_any(ax, fontsize=16, ncol=1, loc="upper left")

        fig.tight_layout()
        out_path = os.path.join(cloud_dir, f"alpha_height_raw_t{time_tag}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
        drag1_split_2d: Optional[Dict[str, np.ndarray]] = None,
        drag1_con_2d: Optional[Dict[str, np.ndarray]] = None,
        velocity_diff_2d: Optional[Dict[str, np.ndarray]] = None,
        extra_vtk_fields: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Save every term as a standalone 2D VTK file for Paraview."""
        time_tag = self._time_tag(time_v)
        vtk_dir = os.path.join(self.output_dir, f"vtk_t{time_tag}")
        os.makedirs(vtk_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)
        print(f"VTK y-axis range: {y_axis.min()} to {y_axis.max()} ")
        y_vals = y_axis / self.H
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

        if drag1_split_2d:
            for name, field in drag1_split_2d.items():
                field_seg = field[: head_idx + 1, :][mask, :]
                out_path = os.path.join(vtk_dir, f"{name}_t{time_tag}.vtk")
                self._write_structured_grid_vtk(out_path, xx, yy, name, field_seg)

        if velocity_diff_2d:
            for name, field in velocity_diff_2d.items():
                field_seg = field[: head_idx + 1, :][mask, :]
                out_path = os.path.join(vtk_dir, f"{name}_t{time_tag}.vtk")
                self._write_structured_grid_vtk(out_path, xx, yy, name, field_seg)
                
        if drag1_con_2d:
            for name, field in drag1_con_2d.items():
                field_seg = field[: head_idx + 1, :][mask, :]
                out_path = os.path.join(vtk_dir, f"{name}_t{time_tag}.vtk")
                self._write_structured_grid_vtk(out_path, xx, yy, name, field_seg)        

        if extra_vtk_fields:
            for name, field in extra_vtk_fields.items():
                field_seg = field[: head_idx + 1, :][mask, :]
                if field_seg.shape != xx.shape:
                    print(f"Skip {name}: expected shape {xx.shape}, got {field_seg.shape}")
                    continue
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
        if not comparison_frames:
            print("No valid data found. Skip multi-variable comparison figure.")
            return

        if not self.save_comparison_png:
            return

        requested_columns = [self._as_curve_column(col) for col in self.comparison_columns]
        if self.num_comparison_variables is not None:
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
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(True, linestyle="--", alpha=0.35)
            self._legend_if_any(ax, fontsize=18, ncol=1, loc="upper left")
            fig.tight_layout()

            fig_path = os.path.join(self.output_dir, f"TKE_Comparison_{col}.png")
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            print(f"Saved Figure: {fig_path}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[pd.DataFrame]:
        time_label = self._time_label(time_v)
        data_3d = self._load_terms_3d(grid, float(time_v))
        if data_3d is None:
            return None

        alpha_a_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            print(f"No alpha.a > threshold ({self.alpha_threshold}) at {time_label}. Skip output.")
            return None

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx}) at {time_label}")

        beta_3d = 1.0 - data_3d.alpha_a

        shear2 = np.square(data_3d.gradUb_ux_dz)
        numer = -9.81 * data_3d.gradMixrho_z
        valid = np.isfinite(numer) & np.isfinite(shear2) & (shear2 > 1e-20)
        Rig = np.zeros_like(numer, dtype=float)
        Rig = np.divide(numer, shear2 * 1000, out=np.full_like(numer, np.nan), where=valid)

        Rig_2d = np.mean(Rig, axis=2)
        
        # 1. 提取全物理量纲(未缩小尺寸)下的 2D 场
        terms_3d = {name: field * beta_3d for name, field in data_3d.terms.items()}
        kdiv_3d = data_3d.kdivub*beta_3d
        kdiv_2d = np.mean(kdiv_3d, axis=2)
        ubx_2d = np.mean(data_3d.Ub[0], axis=2)
        graduxuz = np.mean(data_3d.gradUb_ux_dz, axis=2)
        gradMixrho_z_2d = np.mean(data_3d.gradMixrho_z, axis=2)
        grad_alpha_2d = np.mean(data_3d.grad_alpha, axis=3) if data_3d.grad_alpha is not None else None
        lambda2_2d = np.mean(data_3d.lambda2, axis=2)
        vorticity_x_2d = np.mean(data_3d.ubvorticity_x, axis=2)
        vorticity_y_2d = np.mean(data_3d.ubvorticity_y, axis=2)
        vorticity_z_2d = np.mean(data_3d.ubvorticity_z, axis=2)
        kb_2d = np.mean(data_3d.kb, axis=2)
        coeff_2d = np.mean(data_3d.coeff, axis=2) if data_3d.coeff is not None else None
        self._print_upper_limit_debug(float(time_v), data_3d.x_axis, data_3d.y_axis, ubx_2d)

        terms_2d = self._spanwise_average_terms(terms_3d)
        drag1_split_2d = None
        if data_3d.drag1_split is not None:
            drag1_split_3d = {name: field  for name, field in data_3d.drag1_split.items()}
            drag1_split_2d = self._spanwise_average_terms(drag1_split_3d)
        velocity_diff_2d = None
        if data_3d.velocity_diff is not None:
            velocity_diff_2d = self._spanwise_average_terms(data_3d.velocity_diff)
        drag1_con_2d = None
        if data_3d.drag1_con is not None:
            drag1_con_3d = {name: field for name, field in data_3d.drag1_con.items()}
            drag1_con_2d = self._spanwise_average_terms(drag1_con_3d)
            # terms_2d["drag1_con"] = drag1_con_2d   

        # 2. 直接将有量纲数据传输进底层积分/垂直平均函数中，计算 1D 物理曲线
        df_curve = self._average_to_curves(data_3d.x_axis, data_3d.y_axis, terms_2d, ubx_2d, head_idx, kb_2d,kdiv_2d)
       
        # 3. 【核心变更】在得到 1D dataframe 后，集中进行最后的一步无量纲化
        df_curve = self._nondimensionalize_output_columns(df_curve)
        self._save_alpha_height_cloud(
            float(time_v),
            data_3d.x_axis,
            data_3d.y_axis,
            alpha_a_2d,
            head_idx,
            head_x,
            df_curve,
        )

        # 4. 基于无量纲曲线计算平衡比例项 ratio_avg
        if {"G_avg", "dissipation_avg", "drag1_avg"}.issubset(df_curve.columns):
            numer_ratio = df_curve["drag1_avg"].to_numpy(dtype=float) + df_curve["dissipation_avg"].to_numpy(dtype=float)
            denom_ratio = df_curve["G_avg"].to_numpy(dtype=float)
            ratio = np.divide(
                numer_ratio,
                denom_ratio,
                out=np.zeros_like(denom_ratio, dtype=float),
                where=np.abs(denom_ratio) > 1e-20,
            )
            df_curve["ratio_avg"] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. 为了保持 2D VTK 和云图文件仍然是传统无量纲图，在保存现场生成临时的无量纲副本
        if self.save_vtk:
            terms_2d_dimless = {name: field / (self.U**3 / self.H) for name, field in terms_2d.items()}
            drag1_split_2d_dimless = None
            if drag1_split_2d is not None:
                drag1_split_2d_dimless = {
                    name: field / (self.U**3 / self.H) for name, field in drag1_split_2d.items()
                }
            drag1_con_2d_dimless = None
            if drag1_con_2d is not None:
                drag1_con_2d_dimless = {
                    name: field / (self.U**3 / self.H) for name, field in drag1_con_2d.items()
                }
            velocity_diff_2d_dimless = None
            if velocity_diff_2d is not None:
                velocity_diff_2d_dimless = {name: field / self.U for name, field in velocity_diff_2d.items()}
            kb_2d_dimless = kb_2d / (self.U**2)
            extra_vtk_fields = None
            extra_fields = {}
            if grad_alpha_2d is not None:
                extra_fields["grad_alpha_x"] = grad_alpha_2d[0] * self.H
                extra_fields["grad_alpha_y"] = grad_alpha_2d[1] * self.H
            if coeff_2d is not None:
                extra_fields["drag1_coeff"] = coeff_2d
            if extra_fields:
                extra_vtk_fields = extra_fields
            self._save_2d_vtk(
                time_v=float(time_v),
                x_axis=data_3d.x_axis,
                y_axis=data_3d.y_axis,
                terms_2d=terms_2d_dimless,
                alpha_2d=alpha_a_2d,
                vort_x_2d=vorticity_x_2d,
                vort_y_2d=vorticity_y_2d,
                vort_z_2d=vorticity_z_2d,
                head_idx=head_idx,
                head_x=head_x,
                Rig_2d=Rig_2d,
                gradUb_ux_dz_2d=graduxuz,
                gradMixrho_z_2d=gradMixrho_z_2d,
                lambda2_2d=lambda2_2d,
                ub_2d=ubx_2d,
                kb_2d=kb_2d_dimless,
                drag1_split_2d=drag1_split_2d_dimless,
                drag1_con_2d=drag1_con_2d_dimless,
                velocity_diff_2d=velocity_diff_2d_dimless,
                extra_vtk_fields=extra_vtk_fields,
            )

        # 另外如果后续需要单独调用 _save_2d_clouds 绘图，也请记得传入上文组装好的 terms_2d_dimless
        # self._save_2d_clouds(float(time_v), data_3d.x_axis, data_3d.y_axis, terms_2d_dimless, alpha_a_2d, head_idx, head_x)

        self._save_outputs(float(time_v), df_curve)
        return df_curve

    def run_analysis(self):
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
