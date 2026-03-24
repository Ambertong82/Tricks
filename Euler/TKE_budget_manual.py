import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    omega: np.ndarray
    sus: np.ndarray
    gamma: np.ndarray
    betarhok: np.ndarray
    dbetakdt: np.ndarray


@dataclass
class TimeStepK3D:
    """Container for k field only at one time step (used by dk/dt)."""

    time: float
    betarhok: np.ndarray


class TKEBudgetAnalyzer:
    def __init__(self):
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090311_10"
        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d09_0311_10"
        self.times = [15, 25, 34]

        self.alpha_threshold = 1e-5
        self.rho_w = 1000.0
        self.fig_size = (20, 3)
        self.X_LIM = (0.0, 2.3)
        # self.Y_LIM = (-0.1, 0.005)
        self.curve_lw = 2.0
        # Finite-difference step used for dk/dt. Each target t loads only t-dt, t, t+dt.
        self.dkdt_dt = 0.5

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        """Format float time to match OpenFOAM folder names like 16, 15.5, 16.25."""
        return f"{float(time_v):g}"

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
    def _vertical_average(field2d: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Depth-average along y for each x, producing 1D curve vs x."""
        if hasattr(np, "trapezoid"):
            numerator = np.trapezoid(field2d, x=y_coords, axis=1)
        else:
            numerator = np.trapz(field2d, x=y_coords, axis=1)

        depth = y_coords[-1] - y_coords[0]
        return np.divide(numerator, depth, out=np.zeros_like(numerator), where=np.abs(depth) > 1e-12)

    @staticmethod
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        """Reshape flattened OpenFOAM data after lexsort; reconstruction order is C."""
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

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
                "omega.b": fluidfoam.readscalar(self.sol, time_dir, "omega.b"),
                "alpha.a": fluidfoam.readscalar(self.sol, time_dir, "alpha.a"),
                "alpha.b": fluidfoam.readscalar(self.sol, time_dir, "alpha.b"),
                "SUS": fluidfoam.readscalar(self.sol, time_dir, "SUS"),
                "K": fluidfoam.readscalar(self.sol, time_dir, "K"),
                "dbetakdt": fluidfoam.readscalar(self.sol, time_dir, "dkdt"),
            }
        except Exception as exc:
            print(f"Read failed at t={time_v}: {exc}")
            return None

        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        print(f"Grid shape: nx={nx}, ny={ny}, nz={nz}")

        sort_idx = grid["sort_idx"]
        betarhok = self.rho_w * self._reshape_sorted(raw["alpha.b"], sort_idx, nx, ny, nz) \
            * self._reshape_sorted(raw["k.b"], sort_idx, nx, ny, nz)

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
            omega=self._reshape_sorted(raw["omega.b"], sort_idx, nx, ny, nz),
            sus=self._reshape_sorted(raw["SUS"], sort_idx, nx, ny, nz),
            gamma=self._reshape_sorted(raw["K"], sort_idx, nx, ny, nz),
            betarhok=betarhok,
            dbetakdt = self._reshape_sorted(raw["dbetakdt"], sort_idx, nx, ny, nz),            
        )
    


    def _load_fields_3d_ddt(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepK3D]:
        print(f"\\n>>> Processing time: {time_v}")
        time_dir = self._time_to_dir_name(time_v)

        try:
            raw = {
                "k.b": fluidfoam.readscalar(self.sol, time_dir, "k.b"),
                "alpha.b": fluidfoam.readscalar(self.sol, time_dir, "alpha.b"),
            }
        except Exception as exc:
            print(f"Read failed at t={time_v}: {exc}")
            return None

        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        print(f"Grid shape: nx={nx}, ny={ny}, nz={nz}")
        

        sort_idx = grid["sort_idx"]
        betarhok = self.rho_w * self._reshape_sorted(raw["alpha.b"], sort_idx, nx, ny, nz) \
                * self._reshape_sorted(raw["k.b"], sort_idx, nx, ny, nz)
        
        return TimeStepK3D(
            time=float(time_v),
            betarhok=betarhok
        
        )

    def _compute_dkdt(
        self,
        data_curr: TimeStepFields3D,
        data_prev: Optional[TimeStepK3D],
        data_next: Optional[TimeStepK3D],
    ) -> np.ndarray:
        """Compute time derivative of k using central/one-sided finite difference."""
        if data_prev is not None and data_next is not None:
            dt = data_next.time - data_prev.time
            return self._safe_divide(data_next.betarhok - data_prev.betarhok, dt)

        if data_prev is not None:
            dt = data_curr.time - data_prev.time
            return self._safe_divide(data_curr.betarhok - data_prev.betarhok, dt)

        if data_next is not None:
            dt = data_next.time - data_curr.time
            return self._safe_divide(data_next.betarhok - data_curr.betarhok, dt)

        return np.zeros_like(data_curr.betarhok, dtype=float)

    def _compute_terms_3d(
        self,
        data: TimeStepFields3D,
        data_prev: Optional[TimeStepK3D] = None,
        data_next: Optional[TimeStepK3D] = None,
    ) -> Dict[str, np.ndarray]:
        grad_u = self._gradient_3d(data.Ub, data.x_axis, data.y_axis, data.z_axis)
        grad_alpha = self._gradient_3d(data.alpha_a, data.x_axis, data.y_axis, data.z_axis)
        grad_beta = self._gradient_3d(data.alpha_b, data.x_axis, data.y_axis, data.z_axis)
        hess_beta = self._hessian_3d(data.alpha_b, data.x_axis, data.y_axis, data.z_axis)
        grad_k = self._gradient_3d(data.k, data.x_axis, data.y_axis, data.z_axis)

        
        
        delta_ij = np.eye(3)[:, :, np.newaxis, np.newaxis, np.newaxis]
        s_ij = grad_u + np.transpose(grad_u, axes=(1, 0, 2, 3, 4)) - (2.0/3.0) * np.einsum("ii...->...", grad_u)[np.newaxis, np.newaxis, ...] * delta_ij

        sigma_ij = data.nut * s_ij - (2.0 / 3.0) * data.k * delta_ij
        ## dkdt of term
        betarhokdt = self._compute_dkdt(data, data_prev, data_next)
        dbetarhokdt_ofterm = data.dbetakdt * self.rho_w 

        ## convection
        conv_flux = self.rho_w * data.alpha_b * data.k * data.Ub
        grad_conv_flux = self._gradient_3d(conv_flux, data.x_axis, data.y_axis, data.z_axis)
        convection = np.einsum("ii...->...", grad_conv_flux)

        ## production
        production = self.rho_w * data.alpha_b * np.einsum("ij...,ij...->...", sigma_ij, grad_u)

        ## gradient of density term, involves second derivatives of beta and nonlinear terms of grad beta
        tau_ij = 1e-6 * s_ij
        grad_outer_beta = np.einsum("i...,j...->ij...", grad_beta, grad_beta)
        dd = self._safe_divide(grad_outer_beta, data.alpha_b)
        density_grad_1 = np.einsum("ij...,ij...->...",dd, tau_ij)
        density_grad_2 = np.einsum("ij...,ij...->...", hess_beta, tau_ij)

        sus_safe = np.where(np.abs(data.sus) > 1e-12, data.sus, np.nan)
        omega_safe = np.where(np.abs(data.omega) > 1e-12, data.omega, np.nan)
        beta_safe = np.where(np.abs(data.alpha_b) > 1e-12, data.alpha_b, np.nan)
        

        density_gradient = data.nut  * self.rho_w * (density_grad_1 - density_grad_2)

        ## diffusion term
        coeff = self.rho_w * data.alpha_b * (0.6 * data.nut + 1e-6)   # (nx,ny,nz)
        flux_k = coeff * grad_k                                        # (3,nx,ny,nz)
        grad_flux_k = self._gradient_3d(flux_k, data.x_axis, data.y_axis, data.z_axis)  # (3,3,nx,ny,nz)
        diffusion = np.einsum("ii...->...", grad_flux_k)              # (nx,ny,nz)
        
        ## epsilon  dissipation term
        epsilon = 0.09 * data.k * data.omega
        dissipation = -data.alpha_b * self.rho_w * epsilon
        
        ## drag force terms
        veldiff = -data.Ua + data.Ub
        veldotgradbeta = np.einsum("i...,i...->...", grad_alpha, veldiff)
        drag1 = data.gamma * self._safe_divide(data.nut, sus_safe) * self._safe_divide(veldotgradbeta, beta_safe)

        drag2 = 2.0 * data.gamma * (self._safe_divide(1.0, np.sqrt(sus_safe)) - 1.0) * data.alpha_a * data.k

        coeff3 = (
            2.0
            * data.gamma
            * (self._safe_divide(1.0, np.sqrt(sus_safe)) - 1.0)
            * data.alpha_b
            * data.k
            * self._safe_divide(data.nut, sus_safe)
            * self._safe_divide(np.ones_like(data.omega), omega_safe)
        )
        laplacian_beta = np.einsum("ii...->...", hess_beta)
        drag3 = coeff3 * laplacian_beta



        Summ_right = -convection + production + density_gradient + diffusion + dissipation + drag1 + drag2 + drag3

        return {
            "convection": np.nan_to_num(convection, nan=0.0, posinf=0.0, neginf=0.0),
            "G": np.nan_to_num(production, nan=0.0, posinf=0.0, neginf=0.0),
            "density_gradient": np.nan_to_num(density_gradient, nan=0.0, posinf=0.0, neginf=0.0),
            "dissipation": np.nan_to_num(dissipation, nan=0.0, posinf=0.0, neginf=0.0),
            "dbetarhokdt": np.nan_to_num(dbetarhokdt, nan=0.0, posinf=0.0, neginf=0.0),
            "drag1": np.nan_to_num(drag1, nan=0.0, posinf=0.0, neginf=0.0),
            "drag2": np.nan_to_num(drag2, nan=0.0, posinf=0.0, neginf=0.0),
            "drag3": np.nan_to_num(drag3, nan=0.0, posinf=0.0, neginf=0.0),
            'Summ_right': np.nan_to_num(Summ_right, nan=0.0, posinf=0.0, neginf=0.0),
            'dkdtof': np.nan_to_num(dbetarhokdt_ofterm, nan=0.0, posinf=0.0, neginf=0.0),
            'diff': np.nan_to_num(diffusion, nan=0.0, posinf=0.0, neginf=0.0),
        }

    @staticmethod
    def _spanwise_average_terms(terms_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Average each term along z to obtain unified 2D terms."""
        return {name: np.mean(field, axis=2) for name, field in terms_3d.items()}

    def _average_to_curves(self, x_axis: np.ndarray, y_axis: np.ndarray, terms_2d: Dict[str, np.ndarray], head_idx: int) -> pd.DataFrame:
        x_seg = x_axis[: head_idx + 1]
        x_head = x_axis[head_idx]

        curves = {
            "x": x_seg,
            "x_dime": (x_head - x_seg) / 0.3,
        }

        for name, field in terms_2d.items():
            curve = self._vertical_average(field, y_axis)
            curves[f"{name}_avg"] = curve[: head_idx + 1]

        return pd.DataFrame(curves)

    @staticmethod
    def _format_plot_label(column_name: str) -> str:
        """Map output column names to MathText labels for legends."""
        label_map = {
            "convection_avg": r"$\langle C \rangle_d$",
            "G_avg": r"$\langle G \rangle_d$",
            "density_gradient_avg": r"$\langle D_g \rangle_d$",
            "dissipation_avg": r"$\langle \epsilon \rangle_d$",
            "dbetarhokdt_avg": r"$\langle \partial (\rho\beta k) / \partial t \rangle_d$",
            "drag1_avg": r"$\langle Drag_{d1} \rangle_d$",
            "drag2_avg": r"$\langle Drag_{d2} \rangle_d$",
            "drag3_avg": r"$\langle Drag_{d3} \rangle_d$",
            "Summ_right_avg": r"$\langle RHS \rangle_d$",
            "diff_avg": r"$\langle D \rangle_d$",
        }
        return label_map.get(column_name, column_name)

    def _save_outputs(self, time_v: float, df_curve: pd.DataFrame):
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_v:.2f}.csv")
        df_curve.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        fig, ax = plt.subplots(figsize=self.fig_size)
        for col in df_curve.columns:
            if col in ("x", "x_dime", "dbetarhokdt_avg","dkdtof_avg", "Summ_right_avg"):
                continue
            ax.plot(
                df_curve["x_dime"],
                df_curve[col],
                linewidth=self.curve_lw,
                label=self._format_plot_label(col),
            )

        ax.set_title(f"TKE Budget Terms (Vertical Average) at t={time_v:.2f}s", fontsize=16)
        ax.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax.set_xlim(df_curve["x_dime"].max(), 0.0)
        ax.set_ylabel("Vertical average", fontsize=14)
        # ax.set_ylim(self.Y_LIM)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=10, ncol=2)
        fig.tight_layout()

        fig_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_v:.2f}.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

        fig2, ax2 = plt.subplots(figsize=self.fig_size)
        for col in ("dbetarhokdt_avg", "Summ_right_avg"):
            if col in df_curve.columns:
                ax2.plot(
                    df_curve["x_dime"],
                    df_curve[col],
                    linewidth=self.curve_lw,
                    label=self._format_plot_label(col),
                )

        ax2.set_title(f"TKE Budget Terms summary at t={time_v:.2f}s", fontsize=16)
        ax2.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax2.set_xlim(df_curve["x_dime"].max(), 0.0)
        ax2.set_ylabel("Vertical average", fontsize=14)
        # ax2.set_ylim(self.Y_LIM)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=10, ncol=2)
        fig2.tight_layout()

        fig_path = os.path.join(self.output_dir, f"TKE_Budget_Summary_t{time_v:.2f}.png")
        fig2.savefig(fig_path, dpi=300)
        plt.close(fig2)
        print(f"Saved Figure: {fig_path}")

        fig3, ax3 = plt.subplots(figsize=self.fig_size)
        for col in ("dkdtof_avg", "Summ_right_avg"):
            if col in df_curve.columns:
                ax3.plot(
                    df_curve["x_dime"],
                    df_curve[col],
                    linewidth=self.curve_lw,
                    label=self._format_plot_label(col),
                )

        ax3.set_title(f"TKE Budget Terms summary2 at t={time_v:.2f}s", fontsize=16)
        ax3.set_xlabel(r'$(x_f-x)/H$', fontsize=14)
        ax3.set_xlim(df_curve["x_dime"].max(), 0.0)
        ax3.set_ylabel("Vertical average", fontsize=14)
        # ax3.set_ylim(self.Y_LIM)
        ax3.tick_params(axis="both", labelsize=12)
        ax3.grid(True, linestyle="--", alpha=0.35)
        ax3.legend(fontsize=10, ncol=2)
        fig3.tight_layout()

        fig_path = os.path.join(self.output_dir, f"TKE_Budget_Summary2_t{time_v:.2f}.png")
        fig3.savefig(fig_path, dpi=300)
        plt.close(fig3)
        print(f"Saved Figure: {fig_path}")


    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float):
        t_prev = float(time_v - self.dkdt_dt)
        t_next = float(time_v + self.dkdt_dt)

        data_prev = self._load_fields_3d_ddt(grid, t_prev)
        data_3d = self._load_fields_3d(grid, float(time_v))
        data_next = self._load_fields_3d_ddt(grid, t_next)

        if data_3d is None:
            return

        if data_prev is None and data_next is None:
            print(f"No valid neighbor time step for dk/dt at t={time_v}. Skip output.")
            return

        # Use spanwise-averaged alpha_a only for front/head detection.
        alpha_a_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            print(f"No alpha.a > threshold ({self.alpha_threshold}) at t={time_v}. Skip output.")
            return

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx})")

        terms_3d = self._compute_terms_3d(data_3d, data_prev=data_prev, data_next=data_next)
        terms_2d = self._spanwise_average_terms(terms_3d)
        df_curve = self._average_to_curves(data_3d.x_axis, data_3d.y_axis, terms_2d, head_idx)
        self._save_outputs(float(time_v), df_curve)

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)
        for t in self.times:
            self.process_time_step(grid, t)


if __name__ == "__main__":
    analyzer = TKEBudgetAnalyzer()
    analyzer.run_analysis()
