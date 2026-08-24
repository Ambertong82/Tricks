import os
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm, TwoSlopeNorm


class TurbidityCurrentAnalyzer:
    def __init__(self):
        # OpenFOAM case
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230327_1"
        # self.output_dir = "/home/amber/postpro/u_vorticity/tc3d_23ofcal"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_11"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2test"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_3test"
        self.times = [15,25,35]
        self.output_root = "/home/amber/postpro/u_vorticity"
        self.output_prefix = "tc3d_23"
        

        # 仅读取已经算好的项
        self.vort_fields: Dict[str, str] = {
            "ddt1": "Vort_ddt1",
            "ddt2": "Vort_ddt2",
            "ddt3": "Vort_ddt3",
            "adv1": "Vort_Advection1",
            # "advectioncal1": "Vort_advectioncal1",
            "adv2": "Vort_Advection2",
            "adv3": "Vort_Advection3",
            "adv4": "Vort_Advection4",
            # "advectioncal4": "Vort_Advectioncal4",
            "adv5": "Vort_Advection5",
            "diff1": "Vort_Viscous1",
            "diff2": "Vort_Viscous2",
            "diff3": "Vort_Viscous3",
            "diff4": "Vort_Viscous4",
            # "viscous_diffusioncal4": "Vort_Viscouscal4",
            "diff5": "Vort_Viscous5",
            "gravity1": "Vort_Gravity",
            "pressure1": "Vort_P",
            "drag1": "Vort_Drag1",
            "drag2": "Vort_Drag2",
            "drag3": "Vort_Drag3",
            "vorticityUb": "vorticity_Ub",
        }

        self.fig_size = (30, 12)
        self.cmap = "coolwarm"
        self.n_levels = 121
        self.x_lim = 3.0
        self.y_lim = (0.0, 1.0)
        self.curve_fig_size = (30, 6)
        self.curve_lw = 2.0
        self.alpha_threshold = 1e-5
        self.alpha_interface = 1e-5  # iso-surface level used for interface sampling
        self.head_x_scale = 0.3
        self.clip_negative_x = True
        self.save_curve_csv = True
        self.save_curve_png = False
        # Keep only the whole curves CSV; skip the per-group (TEND/ADV/...)
        # split files.  The user wants "整个的 csv".
        self.save_curve_group_csv = False
        self.curve_groups = {
            r"TEND": ["ddt1", "ddt2", "ddt3", "ddt_sum"],
            r"ADV": ["adv1", "adv2", "adv3", "adv4", "adv5", "adv_sum"],
            r"DIFF": ["diff1", "diff2", "diff3", "diff4", "diff5", "diff_sum"],
            r"DRAG": ["drag1", "drag2", "drag3", "drag_sum"],
            r"GRAVITY": ["gravity1"],
            r"PRESSURE": ["pressure1"],
            # combined gravity + pressure term (added so curves include the sum)
            r"GRAV+P": ["GP"],
        }
        self.curve_sum_groups = {
            "ddt_sum": ["ddt1", "ddt2", "ddt3"],
            "adv_sum": ["adv1", "adv2", "adv3", "adv4", "adv5"],
            "diff_sum": ["diff1", "diff2", "diff3", "diff4", "diff5"],
            "drag_sum": ["drag1", "drag2", "drag3"],
        }
        self.robust_percentile = (1.0, 99.0)
        self.advection_percentile = (3.0, 92.0)
        self.advection_gamma = 0.45
        self.diffusion_percentile = (5.0, 90.0)
        self.diffusion_gamma = 0.35
        self.export_paraview = False
        self.rhoa = 3217
        self.rhob = 1000
        self.time_scale = 1.175
        self.H0 = 0.3
        self.g = 9.81
        self.label_fontsize = 32
        self.title_fontsize = 38
        self.tick_fontsize = 38
        self.legend_fontsize = 32
        self.offset_fontsize = 38

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _reshape_field(field_flat: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        n_cells = nx * ny * nz
        arr = np.asarray(field_flat)

        if sort_idx.size != n_cells:
            raise ValueError(f"sort_idx size mismatch: got {sort_idx.size}, expected {n_cells}")

        if arr.ndim == 1:
            # OpenFOAM uniform scalar fields are returned by fluidfoam as one value.
            if arr.size == 1:
                arr = np.full(n_cells, float(arr[0]), dtype=arr.dtype)
            if arr.size != n_cells:
                raise ValueError(f"field size mismatch: got {arr.size}, expected {n_cells}")
            return arr[sort_idx].reshape(nx, ny, nz)

        if arr.ndim == 2:
            # OpenFOAM uniform vector fields are returned by fluidfoam as one vector.
            if arr.shape == (3, 1):
                arr = np.repeat(arr, n_cells, axis=1)
            elif arr.shape == (1, 3):
                arr = np.repeat(arr.T, n_cells, axis=1)

            # fluidfoam vector layout can be (3, n_cells) or (n_cells, 3)
            if arr.shape == (n_cells, 3):
                arr = arr.T
            if arr.shape == (3, n_cells):
                return arr[:, sort_idx].reshape(3, nx, ny, nz)

        raise ValueError(
            f"Unsupported field shape {arr.shape}; expected (n_cells,), (1,), "
            "(3, n_cells), (n_cells, 3), (3, 1) or (1, 3)"
        )

    @staticmethod
    def compute_spanwise_average(field_3d: np.ndarray) -> np.ndarray:
        # Spanwise direction is the last grid axis (z) after reshape.
        return np.mean(field_3d, axis=-1)

    @staticmethod
    def vector_to_z_component_2d(vector_2d: np.ndarray) -> np.ndarray:
        # Extract z component from a (3, nx, ny) vector field.
        if vector_2d.ndim != 3 or vector_2d.shape[0] != 3:
            raise ValueError(f"Expected vector_2d shape (3, nx, ny), got {vector_2d.shape}")
        return vector_2d[2, :, :]

    @staticmethod
    def _vertical_integral(field_2d: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        if hasattr(np, "trapezoid"):
            return np.trapezoid(field_2d, x=y_coords, axis=1)
        return np.trapz(field_2d, x=y_coords, axis=1)

    @staticmethod
    def vector_to_x_component_2d(vector_2d: np.ndarray) -> np.ndarray:
        # Extract x component from a (3, nx, ny) vector field.
        if vector_2d.ndim != 3 or vector_2d.shape[0] != 3:
            raise ValueError(f"Expected vector_2d shape (3, nx, ny), got {vector_2d.shape}")
        return vector_2d[0, :, :]

    def _vertical_average_to_zerocity_zero(
        self,
        field_2d: np.ndarray,
        y_coords: np.ndarray,
        ubx_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Match the TKE_budget_ofcal logic: average from y>=0.001 up to the first
        # positive-to-nonpositive crossing of Ubx; if no crossing is found, use the top.
        nx = field_2d.shape[0]
        y_lower = 0.001
        out = np.zeros(nx, dtype=float)
        heights = np.zeros(nx, dtype=float)

        for i in range(nx):
            f_profile = field_2d[i]
            u_profile = ubx_2d[i]
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
            out[i] = numerator / height if height > 1e-12 else 0.0

        return out, heights

    def _trim_x_dime(self, x_seg: np.ndarray, x_head: float):
        if self.head_x_scale == 0.0:
            raise ValueError("head_x_scale must be non-zero")
        x_dime = (x_head - x_seg) / self.head_x_scale
        mask = x_dime >= 0.0
        return x_seg[mask], x_dime[mask], mask

    @staticmethod
    def _legend_if_any(ax, **kwargs):
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)
    
    @staticmethod
    def dimensionless_vorticity_transport(q_2d: np.ndarray, density_scale: float, time_scale: float) -> np.ndarray:
        # Vort_* fields include density; divide by density and multiply by T^2.
        if not np.issubdtype(q_2d.dtype, np.floating):
            raise ValueError(f"Expected q_2d to be a floating-point array, got {q_2d.dtype}")
        return q_2d / density_scale * time_scale**2

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

    def _to_head_frame_x(self, x_2d: np.ndarray, head_x: float) -> np.ndarray:
        if self.head_x_scale == 0.0:
            raise ValueError("head_x_scale must be non-zero")
        return (head_x - x_2d) / self.head_x_scale

    def _build_sorted_mesh(self):
        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        nx, ny, nz = len(np.unique(x_raw)), len(np.unique(y_raw)), len(np.unique(z_raw))
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))

        x_3d = self._reshape_field(x_raw, sort_idx, nx, ny, nz)
        y_3d = self._reshape_field(y_raw, sort_idx, nx, ny, nz)

        x_2d = self.compute_spanwise_average(x_3d)
        y_2d = self.compute_spanwise_average(y_3d)
        
        return sort_idx, nx, ny, nz, x_2d, y_2d

    def _build_curve_dataframe(
        self,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        ubx_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ) -> pd.DataFrame:
        x_axis = x_2d[:, 0]
        y_axis = y_2d[0, :]

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)

        curves = {
            "x": x_seg,
            "x_dime": x_dime,
        }

        for name, field_2d in terms_2d.items():
            field_seg = field_2d[: head_idx + 1, :][mask, :]
            curves[f"{name}_avg"], _ = self._vertical_average_to_zerocity_zero(
                field_seg,
                y_axis,
                ubx_2d[: head_idx + 1, :][mask, :],
            )

        return pd.DataFrame(curves)

    def _alpha_threshold_height(self, alpha_2d: np.ndarray, y_coords: np.ndarray, threshold: float) -> np.ndarray:
        """Interpolated y-height where alpha crosses the threshold for each x.

        The interface is the topmost y where alpha >= threshold (concentration
        decreases upward); linear interpolation is used between the two
        bracketing grid points.
        """
        heights = np.full(alpha_2d.shape[0], np.nan, dtype=float)
        for i in range(alpha_2d.shape[0]):
            profile = alpha_2d[i]
            valid = np.isfinite(profile) & np.isfinite(y_coords)
            y_valid = y_coords[valid]
            alpha_valid = profile[valid]

            above = alpha_valid >= threshold
            if y_valid.size == 0 or not np.any(above):
                continue

            top_idx = int(np.where(above)[0].max())
            if top_idx >= y_valid.size - 1:
                heights[i] = float(y_valid[top_idx])
                continue

            a0 = float(alpha_valid[top_idx])
            a1 = float(alpha_valid[top_idx + 1])
            y0 = float(y_valid[top_idx])
            y1 = float(y_valid[top_idx + 1])
            if abs(a1 - a0) > 1e-20:
                heights[i] = y0 + (threshold - a0) / (a1 - a0) * (y1 - y0)
            else:
                heights[i] = y0
        return heights

    @staticmethod
    def _interp_along_y(field_2d: np.ndarray, y_coords: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Linearly interpolate a 2D field at a per-column height h(x)."""
        nx = field_2d.shape[0]
        out = np.full(nx, np.nan, dtype=float)
        for i in range(nx):
            h = heights[i]
            if not np.isfinite(h):
                continue
            profile = field_2d[i]
            valid = np.isfinite(profile) & np.isfinite(y_coords)
            if np.count_nonzero(valid) < 2:
                continue
            out[i] = float(np.interp(h, y_coords[valid], profile[valid]))
        return out

    @staticmethod
    def _alpha_threshold_height_3d(alpha_3d: np.ndarray, y_coords: np.ndarray, threshold: float) -> np.ndarray:
        """Interpolated interface height h(x, z) for every (x, z) column in 3D.

        Same topmost-crossing logic as _alpha_threshold_height, but applied per
        (x, z) vertical column before any spanwise averaging.  Shape (nx, nz);
        NaN where no grid point reaches the threshold.
        """
        nx, ny, nz = alpha_3d.shape
        heights = np.full((nx, nz), np.nan, dtype=float)
        for ix in range(nx):
            for iz in range(nz):
                profile = alpha_3d[ix, :, iz]
                valid = np.isfinite(profile) & np.isfinite(y_coords)
                y_valid = y_coords[valid]
                alpha_valid = profile[valid]
                above = alpha_valid >= threshold
                if y_valid.size == 0 or not np.any(above):
                    continue
                top_idx = int(np.where(above)[0].max())
                if top_idx >= y_valid.size - 1:
                    heights[ix, iz] = float(y_valid[top_idx])
                    continue
                a0 = float(alpha_valid[top_idx])
                a1 = float(alpha_valid[top_idx + 1])
                y0 = float(y_valid[top_idx])
                y1 = float(y_valid[top_idx + 1])
                if abs(a1 - a0) > 1e-20:
                    heights[ix, iz] = y0 + (threshold - a0) / (a1 - a0) * (y1 - y0)
                else:
                    heights[ix, iz] = y0
        return heights

    @staticmethod
    def _interp_along_y_3d(field_3d: np.ndarray, y_coords: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Interpolate a 3D field at a per-(x, z) interface height h(x, z)."""
        nx, ny, nz = field_3d.shape
        out = np.full((nx, nz), np.nan, dtype=float)
        for ix in range(nx):
            for iz in range(nz):
                h = heights[ix, iz]
                if not np.isfinite(h):
                    continue
                profile = field_3d[ix, :, iz]
                valid = np.isfinite(profile) & np.isfinite(y_coords)
                if np.count_nonzero(valid) < 2:
                    continue
                out[ix, iz] = float(np.interp(h, y_coords[valid], profile[valid]))
        return out

    @staticmethod
    def _spanwise_nanmean(arr: np.ndarray) -> np.ndarray:
        """Mean along the last (z) axis ignoring NaN; NaN for all-NaN rows.

        Like np.nanmean(axis=-1) but without the 'Mean of empty slice'
        RuntimeWarning when a whole x-row has no interface (NaN everywhere).
        """
        mask = np.isfinite(arr)
        counts = mask.sum(axis=-1)
        sums = np.where(mask, arr, 0.0).sum(axis=-1)
        out = np.full_like(arr[..., 0], np.nan, dtype=float)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
        return out

    def _build_interface_dataframe(
        self,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        alpha_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ) -> pd.DataFrame:
        """Sample every term at the alpha = alpha_interface iso-surface.

        After spanwise averaging, the iso-surface becomes the contour line
        h(x) where alpha_2d(x, h(x)) = alpha_interface.  Each (spanwise
        averaged) term is linearly interpolated at (x, h(x)) to produce a 1D
        curve of interface values vs x_dime.
        """
        x_axis = x_2d[:, 0]
        y_axis = y_2d[0, :]

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)

        alpha_seg = alpha_2d[: head_idx + 1, :][mask, :]
        heights = self._alpha_threshold_height(alpha_seg, y_axis, self.alpha_interface)

        curves = {
            "x": x_seg,
            "x_dime": x_dime,
            "h_iface": heights,
            "h_iface_H0": heights / self.H0,
        }

        for name, field_2d in terms_2d.items():
            field_seg = field_2d[: head_idx + 1, :][mask, :]
            curves[f"{name}_iface"] = self._interp_along_y(field_seg, y_axis, heights)

        return pd.DataFrame(curves)

    def _build_interface_dataframe_compare(
        self,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        terms_3d: Dict[str, np.ndarray],
        alpha_2d: np.ndarray,
        h_B: np.ndarray,
        head_idx: int,
        head_x: float,
    ) -> pd.DataFrame:
        """Interface values by two orderings, side by side for comparison.

        A) spanwise-average first, then find the alpha = alpha_interface
           contour h(x) and interpolate the averaged terms at (x, h(x)).
        B) find the iso-surface first in 3D: per (x, z) column get h(x, z),
           sample each term on it, then spanwise-average the interface values
           (h_B and the per-term 1D curves in terms_3d are already reduced).

        terms_3d carries the method-B 1D curves for every term (base terms
        sampled on the 3D interface, plus derived sums); NaN where missing.
        """
        x_axis = x_2d[:, 0]
        y_axis = y_2d[0, :]

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)

        # Method A: interface of the spanwise-averaged field
        alpha_seg_A = alpha_2d[: head_idx + 1, :][mask, :]
        h_A = self._alpha_threshold_height(alpha_seg_A, y_axis, self.alpha_interface)

        curves = {
            "x": x_seg,
            "x_dime": x_dime,
            "h_iface_A": h_A,
            "h_iface_A_H0": h_A / self.H0,
            "h_iface_B": h_B,
            "h_iface_B_H0": h_B / self.H0,
        }

        for name, field_2d in terms_2d.items():
            field_seg = field_2d[: head_idx + 1, :][mask, :]
            curves[f"{name}_A"] = self._interp_along_y(field_seg, y_axis, h_A)
            b_val = terms_3d.get(name)
            curves[f"{name}_B"] = b_val if b_val is not None else np.full(x_seg.size, np.nan)

        return pd.DataFrame(curves)

    def _save_interface_compare_outputs(self, time_v: float, df_iface: pd.DataFrame, output_dir: str) -> None:
        if df_iface is None or df_iface.empty:
            return

        time_dir = self._time_to_dir_name(time_v)
        time_dim = float(time_v) * 0.85
        os.makedirs(output_dir, exist_ok=True)

        df_out = df_iface.copy()
        df_out.insert(0, "time", float(time_v))
        df_out.insert(1, "time_dim", time_dim)

        csv_path = os.path.join(output_dir, f"vorticity_interface_{self.alpha_interface}_compare_t{time_dir}.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"  saved: {csv_path}")

    def _save_curve_outputs(self, time_v: float, df_curve: pd.DataFrame, output_dir: str) -> None:
        if not self.save_curve_csv and not self.save_curve_png:
            return

        time_dir = self._time_to_dir_name(time_v)
        # time_v is numeric; compute nondimensional time from numeric value
        time_dim = float(time_v) * 0.85
        os.makedirs(output_dir, exist_ok=True)

        df_curve = df_curve.copy()
        df_curve.insert(0, "time", float(time_v))
        df_curve.insert(1, "time_dim", time_dim)

        if self.save_curve_csv:
            csv_path = os.path.join(output_dir, f"vorticity_curves_t{time_dir}.csv")
            df_curve.to_csv(csv_path, index=False)
            print(f"  saved: {csv_path}")

        plot_mask = df_curve["x_dime"].to_numpy(dtype=float) >= 0.0
        df_plot = df_curve.loc[plot_mask].copy()

        if self.save_curve_group_csv:
            for group_name, short_names in self.curve_groups.items():
                group_cols = ["time", "time_dim", "x", "x_dime"]
                for short_name in short_names:
                    col_name = f"{short_name}_avg"
                    if col_name in df_plot.columns:
                        group_cols.append(col_name)

                if len(group_cols) == 4:
                    continue

                group_df = df_plot[group_cols].copy()
                if self.save_curve_csv:
                    group_csv = os.path.join(output_dir, f"vorticity_curves_{group_name}_t{time_dir}.csv")
                    group_df.to_csv(group_csv, index=False)
                    print(f"  saved: {group_csv}")

    def _read_vector_3d(self, time_dir: str, field_name: str, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        vector_flat = fluidfoam.readvector(self.sol, time_dir, field_name)
        vector_3d = self._reshape_field(vector_flat, sort_idx, nx, ny, nz)
        if vector_3d.ndim != 4 or vector_3d.shape[0] != 3:
            raise ValueError(f"Field {field_name} is not a 3-component vector after reshape: {vector_3d.shape}")
        return vector_3d

    def _read_scalar_3d(self, time_dir: str, field_name: str, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        scalar_flat = fluidfoam.readscalar(self.sol, time_dir, field_name)
        scalar_3d = self._reshape_field(scalar_flat, sort_idx, nx, ny, nz)
        if scalar_3d.ndim != 3:
            raise ValueError(f"Field {field_name} is not scalar after reshape: {scalar_3d.shape}")
        return scalar_3d

    def _plot_contour(
        self,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        q_2d: np.ndarray,
        title: str,
        out_path: str,
        percentile: tuple = None,
        gamma: float = None,
    ) -> None:
        q_plot = np.array(q_2d, copy=True)
        if self.clip_negative_x:
            q_plot = np.where(x_2d < 0.0, np.nan, q_plot)

        q_valid = q_plot[np.isfinite(q_plot)]
        if q_valid.size == 0:
            levels = self.n_levels
            norm = None
        else:
            if percentile is None:
                p_low, p_high = self.robust_percentile
            else:
                p_low, p_high = percentile
            vmin = float(np.percentile(q_valid, p_low))
            vmax = float(np.percentile(q_valid, p_high))

            # Fallback when field is nearly constant.
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(q_valid))
                vmax = float(np.nanmax(q_valid))

            if vmax <= vmin:
                levels = self.n_levels
                norm = None
            elif gamma is not None and gamma > 0.0 and vmin >= 0.0:
                levels = np.linspace(vmin, vmax, self.n_levels)
                norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
            elif vmin < 0.0 < vmax:
                levels = np.linspace(vmin, vmax, self.n_levels)
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                levels = np.linspace(vmin, vmax, self.n_levels)
                norm = None

        plt.figure(figsize=self.fig_size)
        cf = plt.contourf(x_2d, y_2d, q_plot, levels=levels, cmap=self.cmap, norm=norm, extend="both")
        plt.colorbar(cf)

        plt.xlim(0.0, self.x_lim)
        plt.ylim(*self.y_lim)
        plt.xlabel(f"(head_x - x) / {self.head_x_scale}")
        plt.ylabel(f"y / {self.H0}")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def _save_paraview_vtk(self, x_2d: np.ndarray, y_2d: np.ndarray, q_2d: np.ndarray, scalar_name: str, out_path: str) -> None:
        nx, ny = q_2d.shape
        z_2d = np.zeros_like(q_2d)

        with open(out_path, "w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Spanwise averaged field\n")
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
                    val = float(q_2d[i, j])
                    if not np.isfinite(val):
                        val = -9999.0
                    f.write(f"{val:.9e}\n")

    def process_time_step(self, time_v: float, sort_idx: np.ndarray, nx: int, ny: int, nz: int, x_2d: np.ndarray, y_2d: np.ndarray) -> None:
        time_dir = self._time_to_dir_name(time_v)
        output_dir = os.path.join(self.output_root, self.output_prefix)
        paraview_dir = os.path.join(output_dir, f"paraview{time_dir}")
        os.makedirs(output_dir, exist_ok=True)
        if self.export_paraview:
            os.makedirs(paraview_dir, exist_ok=True)

        print(f"Processing t={time_dir}...")

        alpha_a_3d = self._read_scalar_3d(time_dir, "alpha.a", sort_idx, nx, ny, nz)
        ub_3d = self._read_vector_3d(time_dir, "U.b", sort_idx, nx, ny, nz)
        alpha_a_2d = self.compute_spanwise_average(alpha_a_3d)
        ubx_2d = self.vector_to_x_component_2d(self.compute_spanwise_average(ub_3d))
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            head_x = float(np.nanmax(x_2d))
            print(f"  Warning: no alpha.a > {self.alpha_threshold:g}, fallback head_x={head_x:.4g}")
        else:
            head_x = float(x_2d[head_idx, 0])
            print(f"  head_x={head_x:.4g} (idx={head_idx}, threshold={self.alpha_threshold:g})")
        x_plot_2d = self._to_head_frame_x(x_2d, head_x)

        gravity1_2d = None
        pressure1_2d = None
        curve_terms_2d: Dict[str, np.ndarray] = {}

        # Method B (interface-first): interface height h(x, z) per (x, z) column
        # in the 3D alpha field, before any spanwise averaging.
        y_axis = y_2d[0, :]
        x_seg0 = x_2d[:, 0][: head_idx + 1]
        _, _, mask0 = self._trim_x_dime(x_seg0, head_x)
        alpha_seg_3d = alpha_a_3d[: head_idx + 1, :, :][mask0, :, :]
        h_xz = self._alpha_threshold_height_3d(alpha_seg_3d, y_axis, self.alpha_interface)
        h_B = self._spanwise_nanmean(h_xz)  # spanwise-averaged interface height
        interface_3d_terms: Dict[str, np.ndarray] = {}

        for short_name, of_field_name in self.vort_fields.items():
            q_vec_3d = self._read_vector_3d(time_dir, of_field_name, sort_idx, nx, ny, nz)
            q_vec_2d = self.compute_spanwise_average(q_vec_3d)
            q_2d = self.vector_to_z_component_2d(q_vec_2d)
            q_2d = self.dimensionless_vorticity_transport(q_2d, self.rhob, self.time_scale)

            # Method B: sample the z-component 3D field on the per-(x, z)
            # iso-surface, then spanwise-average those interface values.
            q_z_3d = self.dimensionless_vorticity_transport(q_vec_3d[2], self.rhob, self.time_scale)
            q_seg_3d = q_z_3d[: head_idx + 1, :, :][mask0, :, :]
            val_xz = self._interp_along_y_3d(q_seg_3d, y_axis, h_xz)
            interface_3d_terms[short_name] = self._spanwise_nanmean(val_xz)

            if short_name == "gravity1":
                gravity1_2d = q_2d

            elif short_name == "pressure1":
                pressure1_2d = q_2d

            curve_terms_2d[short_name] = q_2d



            if self.export_paraview:
                # Prepare VTK coordinates: y should be normalized by H0, x should be
                # the head-frame (x_plot_2d) clipped to [0, x_lim]. This keeps VTK
                # coordinates consistent with plotted axes.
                x_vtk = np.clip(x_plot_2d, 0.0, self.x_lim)
                y_vtk = y_2d / self.H0

                vtk_name = f"{short_name}_spanwise_t{time_dir}.vtk"
                vtk_path = os.path.join(paraview_dir, vtk_name)
                self._save_paraview_vtk(
                    x_vtk,
                    y_vtk,
                    q_2d,
                    scalar_name=f"{short_name}_z_spanwise",
                    out_path=vtk_path,
                )
                self._save_paraview_vtk(
                    x_vtk,
                    y_vtk,
                    alpha_a_2d,
                    scalar_name="alpha_a_spanwise",
                    out_path=os.path.join(paraview_dir, f"alpha_a_spanwise_t{time_dir}.vtk"),
                )

        if self.export_paraview and gravity1_2d is not None and pressure1_2d is not None:
            gp_sum_2d = gravity1_2d + pressure1_2d
            
            gp_name = f"gravity1_plus_pressure1_spanwise_t{time_dir}.vtk"
            gp_path = os.path.join(paraview_dir, gp_name)
            # For combined field VTK also use clipped head-frame x and normalized y.
            x_vtk = np.clip(x_plot_2d, 0.0, self.x_lim)
            y_vtk = y_2d / self.H0

            self._save_paraview_vtk(
                x_vtk,
                y_vtk,
                gp_sum_2d,
                scalar_name="gravity1_plus_pressure1_z_spanwise",
                out_path=gp_path,
            )
            print(f"  saved: {gp_path}")
 


        curve_terms_for_output = dict(curve_terms_2d)
        if gravity1_2d is not None and pressure1_2d is not None:
            curve_terms_for_output["GP"] = gravity1_2d + pressure1_2d
        for sum_name, term_names in self.curve_sum_groups.items():
            if all(term_name in curve_terms_for_output for term_name in term_names):
                curve_terms_for_output[sum_name] = sum(curve_terms_for_output[term_name] for term_name in term_names)

        # Derived sums for method B (spanwise mean is linear, so the interface
        # value of a sum equals the sum of the interface values).
        for sum_name, term_names in self.curve_sum_groups.items():
            if all(term_name in interface_3d_terms for term_name in term_names):
                interface_3d_terms[sum_name] = sum(interface_3d_terms[term_name] for term_name in term_names)
        if "gravity1" in interface_3d_terms and "pressure1" in interface_3d_terms:
            interface_3d_terms["GP"] = interface_3d_terms["gravity1"] + interface_3d_terms["pressure1"]

        if curve_terms_for_output:
            df_curve = self._build_curve_dataframe(x_2d, y_2d, curve_terms_for_output, ubx_2d, head_idx, head_x)
            self._save_curve_outputs(time_v, df_curve, output_dir)

            # Interface values by both orderings; one CSV per time step.
            df_iface = self._build_interface_dataframe_compare(
                x_2d,
                y_2d,
                curve_terms_for_output,
                interface_3d_terms,
                alpha_a_2d,
                h_B,
                head_idx,
                head_x,
            )
            self._save_interface_compare_outputs(time_v, df_iface, output_dir)

    def run_analysis(self):
        os.makedirs(self.output_root, exist_ok=True)
        sort_idx, nx, ny, nz, x_2d, y_2d = self._build_sorted_mesh()

        for t in self.times:
            self.process_time_step(float(t), sort_idx, nx, ny, nz, x_2d, y_2d)


if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()
