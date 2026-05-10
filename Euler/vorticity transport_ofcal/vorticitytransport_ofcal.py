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
        self.times = [15,25,35]
        self.output_root = "/home/amber/postpro/u_vorticity"
        self.output_prefix = "tc3d_23ofcal"
        

        # 仅读取已经算好的项
        self.vort_fields: Dict[str, str] = {
            "ddt1": "Vort_ddt1",
            "ddt2": "Vort_ddt2",
            "ddt3": "Vort_ddt3",
            "advection1": "Vort_Advection1",
            # "advectioncal1": "Vort_advectioncal1",
            "advection2": "Vort_Advection2",
            "advection3": "Vort_Advection3",
            "advection4": "Vort_Advection4",
            # "advectioncal4": "Vort_Advectioncal4",
            "advection5": "Vort_Advection5",
            "viscous_diffusion1": "Vort_Viscous1",
            "viscous_diffusion2": "Vort_Viscous2",
            "viscous_diffusion3": "Vort_Viscous3",
            "viscous_diffusion4": "Vort_Viscous4",
            # "viscous_diffusioncal4": "Vort_Viscouscal4",
            "viscous_diffusion5": "Vort_Viscous5",
            "gravity1": "Vort_Gravity",
            "pressure1": "Vort_P",
            "drag1": "Vort_Drag1",
            "drag2": "Vort_Drag2",
            "drag3": "Vort_Drag3",
            "vorticityUb": "vorticity_Ub",
        }

        self.fig_size = (20, 6)
        self.cmap = "coolwarm"
        self.n_levels = 121
        self.x_lim = 4.0
        self.y_lim = (0.0, 1.0)
        self.curve_fig_size = (20, 3.5)
        self.curve_lw = 2.0
        self.alpha_threshold = 1e-5
        self.head_x_scale = 0.3
        self.clip_negative_x = True
        self.save_curve_csv = True
        self.save_curve_png = True
        self.curve_groups = {
            "ddt": ["ddt1", "ddt2", "ddt3"],
            "advection": ["advection1", "advection2", "advection3", "advection4", "advection5"],
            "diffusion": ["viscous_diffusion1", "viscous_diffusion2", "viscous_diffusion3", "viscous_diffusion4", "viscous_diffusion5"],
            "drag": ["drag1", "drag2", "drag3"],
            "gravity": ["gravity1"],
            "pressure": ["pressure1"],
        }
        self.robust_percentile = (1.0, 99.0)
        self.advection_percentile = (3.0, 92.0)
        self.advection_gamma = 0.45
        self.diffusion_percentile = (5.0, 90.0)
        self.diffusion_gamma = 0.35
        self.export_paraview = True
        self.rhoa = 3217
        self.rhob = 1000
        self.H0 = 0.3
        self.g = 9.81

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
            if arr.size != n_cells:
                raise ValueError(f"field size mismatch: got {arr.size}, expected {n_cells}")
            return arr[sort_idx].reshape(nx, ny, nz)

        if arr.ndim == 2:
            # fluidfoam vector layout can be (3, n_cells) or (n_cells, 3)
            if arr.shape == (n_cells, 3):
                arr = arr.T
            if arr.shape == (3, n_cells):
                return arr[:, sort_idx].reshape(3, nx, ny, nz)

        raise ValueError(f"Unsupported field shape {arr.shape}; expected (n_cells,), (3, n_cells) or (n_cells, 3)")

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
        mask = (x_dime >= 0.0) & (x_dime <= self.x_lim)
        return x_seg[mask], x_dime[mask], mask

    @staticmethod
    def _legend_if_any(ax, **kwargs):
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)
    
    @staticmethod
    def dimensionless_vorticity(q_2d: np.ndarray, rhoa: float, rhob: float, H0: float, g: float) -> np.ndarray:
        # Convert vorticity to dimensionless form using reference scales.
        # This assumes q_2d has dimensions of 1/s (vorticity).
        if not np.issubdtype(q_2d.dtype, np.floating):
            raise ValueError(f"Expected q_2d to be a floating-point array, got {q_2d.dtype}")
        gstar = g * (rhoa * 0.01 + rhob * 0.99 - rhob) / (rhoa - rhob)
        scale = rhob * gstar / H0
        return q_2d / scale

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
            curves[f"{name}_avg"], curves[f"{name}_height"] = self._vertical_average_to_zerocity_zero(
                field_seg,
                y_axis,
                ubx_2d[: head_idx + 1, :][mask, :],
            )
            curves[f"{name}_integral"] = self._vertical_integral(field_seg, y_axis)

        return pd.DataFrame(curves)

    def _save_curve_outputs(self, time_v: float, df_curve: pd.DataFrame, curve_dir: str) -> None:
        if not self.save_curve_csv and not self.save_curve_png:
            return

        time_dir = self._time_to_dir_name(time_v)
        os.makedirs(curve_dir, exist_ok=True)

        if self.save_curve_csv:
            csv_path = os.path.join(curve_dir, f"vorticity_curves_t{time_dir}.csv")
            df_curve.to_csv(csv_path, index=False)
            print(f"  saved: {csv_path}")

        plot_mask = (df_curve["x_dime"].to_numpy(dtype=float) >= 0.0) & (
            df_curve["x_dime"].to_numpy(dtype=float) <= self.x_lim
        )
        df_plot = df_curve.loc[plot_mask].copy()

        for group_name, short_names in self.curve_groups.items():
            group_cols = ["x", "x_dime"]
            for short_name in short_names:
                for suffix in ("_avg", "_integral", "_height"):
                    col_name = f"{short_name}{suffix}"
                    if col_name in df_plot.columns:
                        group_cols.append(col_name)

            if len(group_cols) == 2:
                continue

            group_df = df_plot[group_cols].copy()
            if self.save_curve_csv:
                group_csv = os.path.join(curve_dir, f"vorticity_curves_{group_name}_t{time_dir}.csv")
                group_df.to_csv(group_csv, index=False)
                print(f"  saved: {group_csv}")

            if not self.save_curve_png:
                continue

            for suffix, title_part, ylabel, file_tag in (
                ("_avg", "Vertical Average", "Average", "avg"),
                ("_integral", "Vertical Integral", "Integral", "int"),
                ("_height", "Selection Height", "Height (m)", "height"),
            ):
                cols = [f"{short_name}{suffix}" for short_name in short_names if f"{short_name}{suffix}" in df_plot.columns]
                if not cols:
                    continue

                fig, ax = plt.subplots(figsize=self.curve_fig_size)
                for col in cols:
                    ax.plot(df_plot["x_dime"], df_plot[col], linewidth=self.curve_lw, label=col)

                ax.set_title(f"{group_name.capitalize()} Terms ({title_part}) at t={time_dir}", fontsize=22)
                ax.set_xlabel(rf"$(x_f-x)/{self.head_x_scale}$", fontsize=20)
                ax.set_xlim(self.x_lim, 0.0)
                ax.set_ylabel(ylabel, fontsize=20)
                ax.tick_params(axis="both", labelsize=18)
                ax.grid(True, linestyle="--", alpha=0.35)
                if suffix != "_height":
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1, 3))
                    offset_text = ax.yaxis.get_offset_text()
                    offset_text.set_fontsize(16)
                self._legend_if_any(ax, fontsize=14, ncol=3, loc="upper left")
                fig.tight_layout()

                out_path = os.path.join(curve_dir, f"vorticity_curves_{group_name}_{file_tag}_t{time_dir}.png")
                fig.savefig(out_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"  saved: {out_path}")

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
        output_dir = os.path.join(self.output_root, f"{self.output_prefix}{time_dir}")
        paraview_dir = os.path.join(output_dir, f"paraview{time_dir}")
        curve_dir = os.path.join(output_dir, f"curve_t{time_dir}")
        os.makedirs(output_dir, exist_ok=True)
        if self.export_paraview:
            os.makedirs(paraview_dir, exist_ok=True)
        
        os.makedirs(curve_dir, exist_ok=True)

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

        for short_name, of_field_name in self.vort_fields.items():
            q_vec_3d = self._read_vector_3d(time_dir, of_field_name, sort_idx, nx, ny, nz)
            q_vec_2d = self.compute_spanwise_average(q_vec_3d)
            q_2d = self.vector_to_z_component_2d(q_vec_2d)
            
            if short_name == "gravity1":
                gravity1_2d = q_2d
                
            elif short_name == "pressure1":
                pressure1_2d = q_2d
                
            curve_terms_2d[short_name] = q_2d



            png_name = f"{short_name}_spanwise_t{time_dir}.png"
            out_path = os.path.join(output_dir, png_name)


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

            print(f"  saved: {out_path}")

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
            curve_terms_for_output["gravity1_plus_pressure1"] = gravity1_2d + pressure1_2d

        if curve_terms_for_output:
            df_curve = self._build_curve_dataframe(x_2d, y_2d, curve_terms_for_output, ubx_2d, head_idx, head_x)
            self._save_curve_outputs(time_v, df_curve, curve_dir)

    def run_analysis(self):
        os.makedirs(self.output_root, exist_ok=True)
        sort_idx, nx, ny, nz, x_2d, y_2d = self._build_sorted_mesh()

        for t in self.times:
            self.process_time_step(float(t), sort_idx, nx, ny, nz, x_2d, y_2d)


if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()