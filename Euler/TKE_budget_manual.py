import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script computes TKE budget terms from raw OF fields without relying on
# pre-computed terms in OpenFOAM.
@dataclass
class TimeStepFields3D:
    """Container for reconstructed 3D fields at one time step."""

    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    Ua: np.ndarray
    Ub: np.ndarray
    alpha_a: np.ndarray
    alpha_b: np.ndarray
    nut: np.ndarray
    k: np.ndarray
    omega: Optional[np.ndarray] = None
    sus: Optional[np.ndarray] = None
    grad_u: Optional[np.ndarray] = None
    grad_alpha_a: Optional[np.ndarray] = None
    grad_alpha_b: Optional[np.ndarray] = None
    grad_k: Optional[np.ndarray] = None
    dbetakdt: Optional[np.ndarray] = None
    gradbetak: Optional[np.ndarray] = None


@dataclass
class TimeStepK3D:
    """Container for k field only at one time step (used by dk/dt)."""

    time: float
    betarhok: np.ndarray


class TKEBudgetAnalyzer:
    def __init__(self):
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_11"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230327_1"
        # self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/3d/case090311_22"
        # self.sol =  "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/NEW/Middle_particle/case230311_2"

        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d09_0327_1conservation"
        # self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0327_1onservation"
        self.times = [15,25,35]

        self.alpha_threshold = 1e-5
        self.rho_w = 1000.0
        self.fig_size = (20, 3)
        self.X_LIM = (0.0, 2.3)
        self.x_dime_max = 4.0
        self.ri_x_dime_max = 4.0
        # self.Y_LIM = (-0.1, 0.005)
        self.curve_lw = 2.0
        # Finite-difference step used for dk/dt. Each target t loads only t-dt, t, t+dt.
        self.dkdt_dt = 0.5
        # grad(U.b) must come from OpenFOAM field; do not fallback to np.gradient.
        self.require_of_grad_u = True
        # Save OF-gradU vs Python-gradU comparison outputs.
        self.enable_grad_u_compare = True
        self.U= 0.26
        self.H = 0.3

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        """Format float time to match OpenFOAM folder names like 16, 15.5, 16.25."""
        return f"{float(time_v):g}"

    @staticmethod
    def _nondim_time(time_v: float) -> float:
        """Convert physical time to the nondimensional time used in outputs."""
        return float(time_v) * 0.85

    def _time_tag(self, time_v: float) -> str:
        """Format nondimensional time for filenames and directories."""
        return f"{self._nondim_time(time_v):.2f}"

    def _time_label(self, time_v: float) -> str:
        """Format nondimensional time for plot titles and legend labels."""
        return rf"$t^*={self._nondim_time(time_v):.2f}$"

    @staticmethod
    def _build_grid_cache(X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Precompute mesh metadata used to reshape flattened OpenFOAM fields."""
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

        X = X_raw[sort_idx].reshape((nx, ny, nz), order="C")
        Y = Y_raw[sort_idx].reshape((nx, ny, nz), order="C")
        Z = Z_raw[sort_idx].reshape((nx, ny, nz), order="C")

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis_3d": X[:, 0, 0],
            "y_axis_3d": Y[0, :, 0],
            "z_axis_3d": Z[0, 0, :],
        }

    @staticmethod
    def _gradient_3d(field_data: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
        """Compute full 3D gradients.

        Scalar input (nx, ny, nz) -> (3, nx, ny, nz)
        Vector input (3, nx, ny, nz) -> (3, 3, nx, ny, nz), as [d/dxj, Ui, ...]
        """
        arr = np.asarray(field_data)

        if arr.ndim == 3:
            grad_x, grad_y, grad_z = np.gradient(arr, x_axis, y_axis, z_axis, axis=(0, 1, 2), edge_order=1)
            return np.stack((grad_x, grad_y, grad_z), axis=0)

        if arr.ndim == 4 and arr.shape[0] == 3:
            grad = np.zeros((3, 3, *arr.shape[1:]), dtype=float)
            for comp in range(3):
                gx, gy, gz = np.gradient(arr[comp], x_axis, y_axis, z_axis, axis=(0, 1, 2), edge_order=1)
                grad[0, comp] = gx
                grad[1, comp] = gy
                grad[2, comp] = gz
            return grad

        raise ValueError(f"Unsupported field shape for _gradient_3d: {arr.shape}")

    @staticmethod
    def _hessian_3d(field_data: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
        """Compute full 3D Hessian for scalar field, returns [i, j, x, y, z]."""
        arr = np.asarray(field_data)
        if arr.ndim != 3:
            raise ValueError(f"_hessian_3d expects scalar field (nx, ny, nz), got {arr.shape}")

        grad = TKEBudgetAnalyzer._gradient_3d(arr, x_axis, y_axis, z_axis)
        hess = np.zeros((3, 3, *arr.shape), dtype=float)

        for i in range(3):
            dg_dx, dg_dy, dg_dz = np.gradient(grad[i], x_axis, y_axis, z_axis, axis=(0, 1, 2), edge_order=1)
            hess[i, 0] = dg_dx
            hess[i, 1] = dg_dy
            hess[i, 2] = dg_dz

        # Enforce symmetry to reduce numerical noise in mixed derivatives.
        return 0.5 * (hess + np.swapaxes(hess, 0, 1))

    @staticmethod
    def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        num_arr, den_arr = np.broadcast_arrays(np.asarray(num, dtype=float), np.asarray(den, dtype=float))
        out = np.zeros_like(den_arr, dtype=float)
        np.divide(num_arr, den_arr, out=out, where=np.abs(den_arr) > 1e-12)
        return out

  
    @staticmethod
    def _vertical_average_to_zerocity_zero(
        field2d: np.ndarray,
        y_coords: np.ndarray,
        ubx2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average along y from the bottom up to the first Ubx zero crossing.

        The cutoff is defined by the first positive-to-nonpositive sign change in Ub[0].
        If no crossing exists, the full available height is used.
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
    
    @staticmethod
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        """Reshape flattened OpenFOAM data after lexsort; reconstruction order is C."""
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    @staticmethod
    def _reshape_vector_gradient(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        """Reshape OpenFOAM vector gradient field to (3, nx, ny, nz)."""
        vec = TKEBudgetAnalyzer._reshape_sorted(field, sort_idx, nx, ny, nz)
        if vec.ndim != 4 or vec.shape[0] != 3:
            raise ValueError(f"Unexpected vector gradient shape: {vec.shape}")
        return vec

    @staticmethod
    def _reshape_tensor_gradient(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        """Reshape OpenFOAM tensor field to (3, 3, nx, ny, nz)."""
        ten = TKEBudgetAnalyzer._reshape_sorted(field, sort_idx, nx, ny, nz)
        if ten.ndim != 4 or ten.shape[0] != 9:
            raise ValueError(f"Unexpected tensor gradient shape: {ten.shape}")
        return ten.reshape((3, 3, nx, ny, nz), order="C")

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        """Find last x index where spanwise-averaged alpha_a exceeds threshold at any y."""
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if len(valid_x) == 0:
            return None
        return int(valid_x.max())

    def _load_fields_3d(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepFields3D]:
        print(f"\\n>>> Processing time: {time_v}")
        time_dir = self._time_to_dir_name(time_v)

        try:
            raw = {
                "U.a": fluidfoam.readvector(self.sol, time_dir, "U.a"),
                "U.b": fluidfoam.readvector(self.sol, time_dir, "U.b"),
                "nut.b": fluidfoam.readscalar(self.sol, time_dir, "nut.b"),
                "k.b": fluidfoam.readscalar(self.sol, time_dir, "k.b"),
                "alpha.a": fluidfoam.readscalar(self.sol, time_dir, "alpha.a"),
                "alpha.b": fluidfoam.readscalar(self.sol, time_dir, "alpha.b"),
                # "K": fluidfoam.readscalar(self.sol, time_dir, "K"),
                "gradbeta": fluidfoam.readvector(self.sol, time_dir, "grad(alpha.b)"),
                "gradU": fluidfoam.readtensor(self.sol, time_dir, "grad(U.b)"),
                "gradalpha": fluidfoam.readvector(self.sol, time_dir, "grad(alpha.a)"),
                "dbetakdt": fluidfoam.readscalar(self.sol, time_dir, "drhokdt"),
                "gradbetak": fluidfoam.readvector(self.sol, time_dir, "gradbetak"),
            }
        except Exception as exc:
            print(f"Read failed at t={time_v}: {exc}")
            return None
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        print(f"Grid shape: nx={nx}, ny={ny}, nz={nz}")

        sort_idx = grid["sort_idx"]

        return TimeStepFields3D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            Ua=self._reshape_sorted(raw["U.a"], sort_idx, nx, ny, nz),
            Ub=self._reshape_sorted(raw["U.b"], sort_idx, nx, ny, nz),
            alpha_a=self._reshape_sorted(raw["alpha.a"], sort_idx, nx, ny, nz),
            alpha_b=self._reshape_sorted(raw["alpha.b"], sort_idx, nx, ny, nz),
            nut=self._reshape_sorted(raw["nut.b"], sort_idx, nx, ny, nz),
            k=self._reshape_sorted(raw["k.b"], sort_idx, nx, ny, nz),

            grad_u=self._reshape_tensor_gradient(raw["gradU"], sort_idx, nx, ny, nz) if self.require_of_grad_u else None,
            grad_alpha_a=self._reshape_vector_gradient(raw["gradalpha"], sort_idx, nx, ny, nz),
            grad_alpha_b=self._reshape_vector_gradient(raw["gradbeta"], sort_idx, nx, ny, nz),
            grad_k=None,
            dbetakdt=self._reshape_sorted(raw["dbetakdt"], sort_idx, nx, ny, nz),
            gradbetak=self._reshape_vector_gradient(raw["gradbetak"], sort_idx, nx, ny, nz)
             
        )

    def _load_fields_3d_ddt(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepK3D]:
        print(f"\\n>>> Processing time: {time_v}")
        time_dir = self._time_to_dir_name(time_v)

        try:
            raw = {
                "k.b": fluidfoam.readscalar(self.sol, time_dir, "k.b"),
                "alpha.b": fluidfoam.readscalar(self.sol, time_dir, "alpha.b"),
                "dbetakdt": fluidfoam.readscalar(self.sol, time_dir, "dbetakdt"),
            }
        except Exception as exc:
            print(f"Read failed at t={time_v}: {exc}")
            return None
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        print(f"Grid shape: nx={nx}, ny={ny}, nz={nz}")

        sort_idx = grid["sort_idx"]

        return TimeStepK3D(
            time=float(time_v),
            betarhok=self._reshape_sorted(raw["k.b"], sort_idx, nx, ny, nz),
            dbetakdt=self._reshape_sorted(raw["dbetakdt"], sort_idx, nx, ny, nz)
        )



    def _compute_terms_3d(
        self,
        data: TimeStepFields3D,
        data_prev: Optional[TimeStepK3D] = None,
        data_next: Optional[TimeStepK3D] = None,
    ) -> Dict[str, np.ndarray]:
        if data.grad_u is None:
            raise ValueError("grad(U.b) is required but not available for this time step.")
        
        
        s_ij = data.grad_u + np.transpose(data.grad_u, axes=(1, 0, 2, 3, 4))

        sigma_ij = data.nut * s_ij
        # Convection term.
        convection1 = data.gradbetak 
        convection11 = np.einsum("i...,i...->...", data.Ub, convection1) if data.Ub is not None else np.zeros_like(data.k)
        conv_flux =  data.alpha_b * data.k
        divU = np.einsum("ii...->...", data.grad_u)
        convection2_dimless = divU * conv_flux /(self.U**3/self.H)
        

        # # k production.
        # production = self.rho_w * data.alpha_b * np.einsum("ij...,ij...->...", sigma_ij, data.grad_u)

        # # Richardson number.
        # numerator = -2217.0 * 9.81 * data.grad_alpha_a[1, ...]
        # denominator = 1000.0 * data.grad_u[1, 0, ...] ** 2
        # Rig = self._safe_divide(numerator, denominator)

        # # TKE
        # k = data.k

        # dbetarhokdt = data.dbetakdt 
        material_derivative_dimless = (data.dbetakdt  + convection11)/ (self.U**3/self.H)
        LHS = material_derivative_dimless + convection2_dimless



        return {
            "convection2": np.nan_to_num(convection2_dimless, nan=0.0, posinf=0.0, neginf=0.0),
            # "dbetarhokdt": np.nan_to_num(dbetarhokdt, nan=0.0, posinf=0.0, neginf=0.0),
            # "G": np.nan_to_num(production, nan=0.0, posinf=0.0, neginf=0.0),
            # "Ri": np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0),
            # "k": np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0),
            "convection11": np.nan_to_num(convection11, nan=0.0, posinf=0.0, neginf=0.0),
            "material_derivative": np.nan_to_num(material_derivative_dimless, nan=0.0, posinf=0.0, neginf=0.0),
            "LHS": np.nan_to_num(LHS, nan=0.0, posinf=0.0, neginf=0.0),
        }

    @staticmethod
    def _spanwise_average_terms(terms_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Average each term along z to obtain unified 2D terms."""
        return {name: np.mean(field, axis=2) for name, field in terms_3d.items()}

    def _average_to_curves(
        self,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        ubx_2d: np.ndarray,
        head_idx: int,
    ) -> pd.DataFrame:
        x_seg = x_axis[: head_idx + 1]
        x_head = x_axis[head_idx]

        curves = {
            "x": x_seg,
            "x_dime": (x_head - x_seg) / 0.3,
        }

        for name, field in terms_2d.items():
            curve, heights = self._vertical_average_to_zerocity_zero(field, y_axis, ubx_2d)
            curves[f"{name}_avg"] = curve[: head_idx + 1]
            curves["height"] = heights[: head_idx + 1]

        return pd.DataFrame(curves)
    
    def _trim_x_dime(self, x_seg: np.ndarray, x_head: float):
        """Trim x-axis segment so x_dime stays within [0, x_dime_max]."""
        x_dime = (x_head - x_seg) / 0.3
        mask = (x_dime >= 0.0) & (x_dime <= self.x_dime_max)
        return x_seg[mask], x_dime[mask], mask

    def _time_png_dir(self, time_v: float) -> str:
        return os.path.join(self.output_dir, f"png_t{self._time_tag(time_v)}")

    @staticmethod
    def _save_figure(fig, fig_path: str):
        """Save a figure with tight bounding box to avoid clipped tick labels."""
        fig.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0.15)

    @staticmethod
    def _format_plot_label(column_name: str) -> str:
        """Map output column names to MathText labels for legends."""
        label_map = {
            "convection2_avg": r"$\langle C2^{*} \rangle_d$",
            "G_avg": r"$\langle G \rangle_d$",
            "Ri_avg": r"$\langle Ri \rangle_d$",
            "density_gradient_avg": r"$\langle D_g \rangle_d$",
            "dissipation_avg": r"$\langle \epsilon \rangle_d$",
            "dbetarhokdt_avg": r"$\langle \partial (\rho\beta k) / \partial t \rangle_d$",
            "k_avg": r"$\langle k \rangle_d$",
            "convection11_avg": r"$\langle C1^{*} \rangle_d$",
            "material_derivative_avg": r"$\langle D(\beta\rho k)/Dt \rangle_d$",
            'LHS_avg': r"$\langle LHS \rangle_d$",
            
        }
        return label_map.get(column_name, column_name)

    def _save_outputs(self, time_v: float, df_curve: pd.DataFrame):
        os.makedirs(self.output_dir, exist_ok=True)
        png_dir = self._time_png_dir(time_v)
        os.makedirs(png_dir, exist_ok=True)

        time_tag = self._time_tag(time_v)
        time_label = self._time_label(time_v)

        csv_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_tag}.csv")
        df_curve.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        fig, ax = plt.subplots(figsize=self.fig_size)
        for col in df_curve.columns:
            if col in ("x", "x_dime",  "LHS_avg"):
                continue
            ax.plot(
                df_curve["x_dime"],
                df_curve[col],
                linewidth=self.curve_lw,
                label=self._format_plot_label(col),
            )

        ax.set_title(f"TKE Budget LHS Terms (Vertical Average) at {time_label}", fontsize=16)
        ax.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax.set_xlim(self.x_dime_max, 0.0)
        ax.set_ylabel("Vertical average", fontsize=14)
        # ax.set_ylim(self.Y_LIM)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)
        if ax.lines:
            ax.legend(fontsize=10, ncol=2)
        fig.tight_layout()

        fig_path = os.path.join(png_dir, f"TKE_Budget_LHS_VAvg_t{time_tag}.png")
        self._save_figure(fig, fig_path)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

        fig2, ax2 = plt.subplots(figsize=self.fig_size)
        for col in ("convection2_avg", "material_derivative_avg", ):
            if col in df_curve.columns:
                ax2.plot(
                    df_curve["x_dime"],
                    df_curve[col],
                    linewidth=self.curve_lw,
                    label=self._format_plot_label(col),
                )

        ax2.set_title(f"Material Derivative and extra Convection Terms at {time_label}", fontsize=16)
        ax2.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax2.set_xlim(self.x_dime_max, 0.0)
        ax2.set_ylabel("Vertical average", fontsize=14)
        # ax2.set_ylim(self.Y_LIM)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.grid(True, linestyle="--", alpha=0.35)
        if ax2.lines:
            ax2.legend(fontsize=10, ncol=2)
        fig2.tight_layout()

        fig_path = os.path.join(png_dir, f"Material_Derivative_Convection_t{time_tag}.png")
        self._save_figure(fig2, fig_path)
        plt.close(fig2)
        print(f"Saved Figure: {fig_path}")

        fig3, ax3 = plt.subplots(figsize=self.fig_size)
        if "convection2_avg" in df_curve.columns:
            ax3.plot(
                df_curve["x_dime"],
                df_curve["convection2_avg"],
                linewidth=self.curve_lw,
                label=self._format_plot_label("convection2_avg"),
            )

        ax3.set_title(f"Convection Term (divergence) at {time_label}", fontsize=16)
        ax3.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax3.set_xlim(self.x_dime_max, 0.0)
        ax3.set_ylabel("Vertical average", fontsize=14)
        # ax3.set_ylim(self.Y_LIM)
        ax3.tick_params(axis="both", labelsize=12)
        ax3.grid(True, linestyle="--", alpha=0.35)
        if ax3.lines:
            ax3.legend(fontsize=10, ncol=2)
        fig3.tight_layout()

        fig_path = os.path.join(png_dir, f"TKE_Budget_Convection_t{time_tag}.png")
        self._save_figure(fig3, fig_path)
        plt.close(fig3)
        print(f"Saved Figure: {fig_path}")

        # Figure 4: G-only curve for focused inspection.
        fig4, ax4 = plt.subplots(figsize=self.fig_size)
        if "G_avg" in df_curve.columns:
            ax4.plot(
                df_curve["x_dime"],
                df_curve["G_avg"],
                linewidth=self.curve_lw,
                label=self._format_plot_label("G_avg"),
            )

        ax4.set_title(f"G Term (Vertical Average) at {time_label}", fontsize=16)
        ax4.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax4.set_xlim(self.x_dime_max, 0.0)
        ax4.set_ylabel("Vertical average", fontsize=14)
        ax4.tick_params(axis="both", labelsize=12)
        ax4.grid(True, linestyle="--", alpha=0.35)
        if ax4.lines:
            ax4.legend(fontsize=10, ncol=1)
        fig4.tight_layout()

        fig4_path = os.path.join(png_dir, f"TKE_Budget_G_only_t{time_tag}.png")
        self._save_figure(fig4, fig4_path)
        plt.close(fig4)
        print(f"Saved Figure: {fig4_path}")

        fig5, ax5 = plt.subplots(figsize=self.fig_size)
        if "k_avg" in df_curve.columns:
            ax5.plot(
                df_curve["x_dime"],
                df_curve["k_avg"],
                linewidth=self.curve_lw,
                label=self._format_plot_label("k_avg"),
            )
        ax5.set_title(f"k (Vertical Average) at {time_label}", fontsize=16)
        ax5.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax5.set_xlim(self.x_dime_max, 0.0)
        ax5.set_ylabel("Vertical average", fontsize=14)
        ax5.tick_params(axis="both", labelsize=12)
        ax5.grid(True, linestyle="--", alpha=0.35)
        if ax5.lines:
            ax5.legend(fontsize=10, ncol=1)
        fig5.tight_layout()

        fig5_path = os.path.join(png_dir, f"TKE_Budget_k_only_t{time_tag}.png")
        self._save_figure(fig5, fig5_path)
        plt.close(fig5)
        print(f"Saved Figure: {fig5_path}")

        if "LHS_avg" in df_curve.columns:
            fig6, ax6 = plt.subplots(figsize=self.fig_size)
            ax6.plot(
                df_curve["x_dime"],
                df_curve["LHS_avg"],
                linewidth=self.curve_lw,
                label=self._format_plot_label("LHS_avg"),
            )
            ax6.set_title(f"LHS of TKE Budget (Vertical Average) at {time_label}", fontsize=16)
            ax6.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
            ax6.set_xlim(self.x_dime_max, 0.0)
            ax6.set_ylabel("Vertical average", fontsize=14)
            ax6.tick_params(axis="both", labelsize=12)
            ax6.grid(True, linestyle="--", alpha=0.35)
            if ax6.lines:
                ax6.legend(fontsize=10, ncol=1)
            fig6.tight_layout()

            fig6_path = os.path.join(png_dir, f"TKE_Budget_LHS_only_t{time_tag}.png")
            self._save_figure(fig6, fig6_path)
            plt.close(fig6)
            print(f"Saved Figure: {fig6_path}")

    def _save_comparison_figure(
        self,
        comparison_frames,
        column_name: str,
        title: str,
        file_name: str,
    ):
        """Save one comparison figure for a single curve column across time steps."""
        if not comparison_frames:
            print(f"No valid data found. Skip {column_name} comparison figure.")
            return

        fig, ax = plt.subplots(figsize=self.fig_size)
        plotted = False

        for time_v, df_curve in comparison_frames:
            if column_name not in df_curve.columns:
                continue

            xvals = df_curve["x_dime"].to_numpy(dtype=float)
            yvals = df_curve[column_name].to_numpy(dtype=float)
            mask = (xvals >= 0.0) & (xvals <= self.x_dime_max)
            if not np.any(mask):
                continue

            ax.plot(
                xvals[mask],
                yvals[mask],
                linewidth=self.curve_lw,
                label=self._time_label(time_v),
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            print(f"No {column_name} curves available. Skip comparison figure.")
            return

        ax.set_title(title, fontsize=22)
        ax.set_xlabel(r'$(x_f-x)/H$', fontsize=20)
        ax.set_xlim(self.x_dime_max, 0.0)
        # ax.set_ylabel("Vertical average", fontsize=14)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=14, ncol=1,loc='upper left')
        #scilimits=(-2, 3) 表示：数量级小于 10^-2 (即0.01) 或大于 10^3 时触发统一的科学计数法
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(16)
        fig.tight_layout()

        fig_path = os.path.join(self.output_dir, file_name)
        self._save_figure(fig, fig_path)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

    def _save_g_comparison(self, comparison_frames):
        self._save_comparison_figure(
            comparison_frames,
            column_name="G_avg",
            title="k Production (G) Comparison Across Time Steps",
            file_name="TKE_Budget_G_Comparison.png",
        )

    def _save_k_comparison(self, comparison_frames):
        self._save_comparison_figure(
            comparison_frames,
            column_name="k_avg",
            title="k Comparison Across Time Steps",
            file_name="TKE_Budget_k_Comparison.png",
        )

    def _save_c2_comparison(self, comparison_frames):
        self._save_comparison_figure(
            comparison_frames,
            column_name="convection2_avg",
            title=r"$\langle C2^{*} \rangle_d$",
            file_name="TKE_Budget_C2_Comparison.png",
        )

    def _save_dmaterialdt_comparison(self, comparison_frames):
        self._save_comparison_figure(
            comparison_frames,
            column_name="material_derivative_avg",
            title=r"$\langle d\phi^*/dt \rangle_d$",
            file_name="TKE_Budget_dMaterialDt_Comparison.png",
        )

    def _save_ri_cloud_plot(self, time_v: float, x_axis: np.ndarray, head_x: float, y_axis: np.ndarray, ri_2d: np.ndarray):
        """Save a spanwise-averaged Ri contour plot in the x-y plane."""
        png_dir = self._time_png_dir(time_v)
        os.makedirs(png_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_dime = (head_x - x_axis) / 0.3
        ri_mask = (x_dime >= 0.0) & (x_dime <= self.ri_x_dime_max)
        x_dime_seg = x_dime[ri_mask]
        x_grid, y_grid = np.meshgrid(x_dime_seg, y_axis, indexing="ij")
        ri_plot = np.ma.masked_invalid(np.clip(ri_2d[ri_mask, :], 0.0, 1.0))

        levels = np.linspace(0.0, 1.0, 51)
        contour = ax.contourf(x_grid, y_grid, ri_plot, levels=levels, cmap="viridis")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Ri")
        cbar.set_ticks(np.linspace(0.0, 1.0, 6))

        ax.set_title(f"Ri Cloud Plot after Spanwise Average at {self._time_label(time_v)}", fontsize=14)
        ax.set_xlabel(r"$(x_f-x)/H$", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(self.ri_x_dime_max, 0.0)
        ax.set_ylim(float(np.min(y_axis)), float(np.max(y_axis)))
        fig.tight_layout()

        fig_path = os.path.join(png_dir, f"Ri_cloud_spanavg_t{self._time_tag(time_v)}.png")
        self._save_figure(fig, fig_path)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

    def _grad_u_curve_components(
        self,
        grad_u: np.ndarray,
        y_axis: np.ndarray,
        ubx_2d: np.ndarray,
        head_idx: int,
    ) -> Dict[str, np.ndarray]:
        """Convert gradU tensor (3,3,nx,ny,nz) to 1D x-curves after z/y averaging."""
        seg = grad_u[:, :, : head_idx + 1, :, :]
        seg_2d = np.mean(seg, axis=4)  # (3,3,nx,ny)

        axis_name = ("x", "y", "z")
        comp = {}
        for i in range(3):
            for j in range(3):
                key = f"dU{axis_name[i]}_d{axis_name[j]}"
                comp[key] = self._vertical_average_to_zerocity_zero(seg_2d[i, j], y_axis, ubx_2d)

        norm_sq = np.zeros_like(next(iter(comp.values())))
        for val in comp.values():
            norm_sq += val * val
        comp["frob_norm"] = np.sqrt(norm_sq)
        return comp

    def _save_grad_u_compare(
        self,
        time_v: float,
        x_axis: np.ndarray,
        head_idx: int,
        head_x: float,
        y_axis: np.ndarray,
        ubx_2d: np.ndarray,
        grad_u_of: np.ndarray,
        grad_u_py: np.ndarray,
    ):
        """Save direct gradU comparison between OF field and Python gradient."""
        png_dir = self._time_png_dir(time_v)
        os.makedirs(png_dir, exist_ok=True)
        x_seg = x_axis[: head_idx + 1]
        x_dime = (head_x - x_seg) / 0.3

        of_comp = self._grad_u_curve_components(grad_u_of, y_axis, ubx_2d, head_idx)
        py_comp = self._grad_u_curve_components(grad_u_py, y_axis, ubx_2d, head_idx)

        compare_data = {"x_dime": x_dime}
        for key in of_comp.keys():
            of_key = f"{key}_of"
            py_key = f"{key}_py"
            d_key = f"{key}_diff"
            compare_data[of_key] = of_comp[key]
            compare_data[py_key] = py_comp[key]
            compare_data[d_key] = py_comp[key] - of_comp[key]

        compare_df = pd.DataFrame(compare_data)
        denom = np.maximum(np.abs(compare_df["frob_norm_of"].to_numpy()), 1e-12)
        compare_df["frob_rel_err"] = np.abs(compare_df["frob_norm_diff"].to_numpy()) / denom

        csv_path = os.path.join(self.output_dir, f"gradU_compare_t{self._time_tag(time_v)}.csv")
        compare_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(compare_df["x_dime"], compare_df["frob_norm_of"], linewidth=self.curve_lw, label="||gradU||_F (OF)")
        ax.plot(compare_df["x_dime"], compare_df["frob_norm_py"], linewidth=self.curve_lw, linestyle="--", label="||gradU||_F (Python)")
        ax.plot(compare_df["x_dime"], np.abs(compare_df["frob_norm_diff"]), linewidth=self.curve_lw, linestyle=":", label="|Delta ||gradU||_F|")
        ax.set_title(f"gradU Comparison: OF vs Python at {self._time_label(time_v)}", fontsize=16)
        ax.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax.set_xlim(self.x_dime_max, 0.0)
        ax.set_ylabel("Vertical average", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=10, ncol=1)
        fig.tight_layout()

        fig_path = os.path.join(png_dir, f"gradU_compare_t{self._time_tag(time_v)}.png")
        self._save_figure(fig, fig_path)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")


    def _save_2d_vtk(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        alpha_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ):
        """Save every term as a standalone 2D VTK file for Paraview."""
        vtk_dir = os.path.join(self.output_dir, f"vtk_t{self._time_tag(time_v)}")
        os.makedirs(vtk_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_seg, x_dime, mask = self._trim_x_dime(x_seg, head_x)
        x_seg_full = x_axis[: head_idx + 1]
        x_dime_full = (head_x - x_seg_full) / 0.3
        ri_mask = (x_dime_full >= 0.0) & (x_dime_full <= self.ri_x_dime_max)
        y_vals = y_axis
        xx, yy = np.meshgrid(x_dime, y_vals, indexing="ij")
        xx_ri, yy_ri = np.meshgrid(x_dime_full[ri_mask], y_vals, indexing="ij")

        alpha_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)[mask, :]
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"alpha_a_t{self._time_tag(time_v)}.vtk"),
            xx,
            yy,
            "alpha_a",
            alpha_seg,
        )
        self._write_structured_grid_vtk(
            os.path.join(vtk_dir, f"alpha_minus_threshold_t{self._time_tag(time_v)}.vtk"),
            xx,
            yy,
            "alpha_minus_threshold",
            alpha_seg - self.alpha_threshold,
        )

        for name, field in terms_2d.items():
            if name == "Ri":
                field_seg = np.clip(field[: head_idx + 1, :][ri_mask, :], 0.0, 1.0)
                out_path = os.path.join(vtk_dir, f"{name}_t{self._time_tag(time_v)}.vtk")
                self._write_structured_grid_vtk(out_path, xx_ri, yy_ri, name, field_seg)
            else:
                field_seg = field[: head_idx + 1, :][mask, :]
                out_path = os.path.join(vtk_dir, f"{name}_t{self._time_tag(time_v)}.vtk")
                self._write_structured_grid_vtk(out_path, xx, yy, name, field_seg)


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

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[pd.DataFrame]:
        # t_prev = float(time_v - self.dkdt_dt)
        # t_next = float(time_v + self.dkdt_dt)

        # data_prev = self._load_fields_3d_ddt(grid, t_prev)
        data_3d = self._load_fields_3d(grid, float(time_v))
        if data_3d is None:
            return None

        alpha_a_2d = np.mean(data_3d.alpha_a, axis=2)
        ubx_2d = np.mean(data_3d.Ub[0], axis=2)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            print(f"No alpha.a > threshold ({self.alpha_threshold}) at {self._time_label(time_v)}. Skip output.")
            return

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx}) at {self._time_label(time_v)}")

        terms_3d= self._compute_terms_3d(data_3d, data_prev=None, data_next=None)
        terms_2d = self._spanwise_average_terms(terms_3d)
        if "Ri" in terms_2d:
            self._save_ri_cloud_plot(float(time_v), data_3d.x_axis, head_x, data_3d.y_axis, terms_2d["Ri"])
        df_curve = self._average_to_curves(data_3d.x_axis, data_3d.y_axis, terms_2d, ubx_2d, head_idx)
        self._save_outputs(float(time_v), df_curve)
        self._save_2d_vtk(float(time_v), data_3d.x_axis, data_3d.y_axis, terms_2d, alpha_a_2d, head_idx, head_x)

        # if self.enable_grad_u_compare:
        #     if data_3d.grad_u is None:
        #         print(f"Skip gradU comparison at t={time_v}: OF gradU not available.")
        #         return
        #     grad_u_py = self._gradient_3d(data_3d.Ub, data_3d.x_axis, data_3d.y_axis, data_3d.z_axis)
        #     self._save_grad_u_compare(
        #         float(time_v),
        #         data_3d.x_axis,
        #         head_idx,
        #         head_x,
        #         data_3d.y_axis,
        #         ubx_2d,
        #         data_3d.grad_u,
        #         grad_u_py,
        #     )

        return df_curve

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        comparison_frames: list[tuple[float, pd.DataFrame]] = []
        for t in self.times:
            df_curve = self.process_time_step(grid, t)
            if df_curve is None:
                continue
            comparison_frames.append((float(t), df_curve))

        self._save_g_comparison(comparison_frames)
        self._save_k_comparison(comparison_frames)
        self._save_c2_comparison(comparison_frames)
        self._save_dmaterialdt_comparison(comparison_frames)


if __name__ == "__main__":
    analyzer = TKEBudgetAnalyzer()
    analyzer.run_analysis()
