#!/usr/bin/env python3
"""
Volume-integrated TKE budget terms for turbidity currents — power and
cumulative work.

Reads OpenFOAM raw fields into a cache, then computes domain-volume integrals:

  εb0       = ∫Ω magUb  dV                      (OF 中已含 α_f·ρ_f·2ν_b)
  εa0       = ∫Ω magUa  dV                      (OF 中已含 α_s·ρ_s·2ν_a)
  εbt       = ∫Ω 0.09·k·ω·α_f·ρ_f  dV          (k-ω dissipation)
  εat       = ∫Ω 2·ν_tb·α_s·ρ_s/SUS·TurThird  dV
  ε_drag01  = ∫Ω γ·α_s·|U_b−U_a|²  dV
  ε_drag02  = ∫Ω γ·ν_tb/(SUS+α_f)·∇α_a·(U_b−U_a)  dV
  ε_dragt1  = ∫Ω 2·γ·α_s·(1/√SUS−1)·k  dV
  ε_dragt2  = ∫Ω γ·α_f·(1/√SUS−1)·ρ_f·k·ν_tb/(ω·SUS)·TurThird  dV

All raw fields are read once per time step and cached to avoid re-reading.

Requires:
  - ``writeCellVolumes`` utility run so that ``0/V`` exists.
  - fluidfoam (pip-installable).

Mesh / I/O conventions follow the sibling script ``energy_terms.py``.
"""

import os
from typing import Dict

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


class TKEIntegralAnalyzer:
    """Full-domain volume integrals of TKE budget terms."""

    # ------------------------------------------------------------------
    # User-configurable parameters
    # ------------------------------------------------------------------
    def __init__(self):
        # OpenFOAM case directory
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_2"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090704_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2"
       
        # Output
        self.output_dir = "/home/amber/postpro/energy_terms_turbulent"
        self.output_prefix = "tc3d_09"

        # Time steps
        self.times = np.arange(0.5, 35.5, 0.5).tolist()

        # Physical parameters
        self.rho_s = 3217.0
        self.rho_f = 1000.0
        self.g = 9.81

        # Reference scales
        self.U_ref = 0.26
        self.H_ref = 0.3
        self.V_ref = 0.0117

        # Output terms (all computed from cached OF fields)
        self.output_terms = [
            "epsilonb0", "epsilona0",
            "epsilonbt", "epsilonat",
            "epsilondrag01", "epsilondrag02",
            "epsilondragt1", "epsilondragt2",
        ]

        # When True, also write non-dimensional columns
        self.nondimensionalize = False

        # Plotting style
        self.fig_size = (10, 6)
        self.title_fontsize = 16
        self.label_fontsize = 14
        self.tick_fontsize = 12
        self.legend_fontsize = 12
        self.marker = "o"
        self.lw = 2.0

    # ══════════════════════════════════════════════════════════════════
    #  Static / helper methods
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _time_tag(time_v: float) -> str:
        return f"{time_v:.2f}"

    @staticmethod
    def _reshape_field(
        field_flat: np.ndarray,
        sort_idx: np.ndarray,
        nx: int,
        ny: int,
        nz: int,
    ) -> np.ndarray:
        """Reshape flat OpenFOAM field → structured (nx, ny, nz)."""
        n_cells = nx * ny * nz
        arr = np.asarray(field_flat)

        if sort_idx.size != n_cells:
            raise ValueError(
                f"sort_idx size mismatch: got {sort_idx.size}, expected {n_cells}"
            )

        # --- scalar ---
        if arr.ndim == 0:
            arr = np.full(n_cells, float(arr.flat[0]), dtype=arr.dtype)
        elif arr.ndim == 1:
            if arr.size == 1:
                arr = np.full(n_cells, float(arr.flat[0]), dtype=arr.dtype)
            if arr.size != n_cells:
                raise ValueError(
                    f"scalar field size mismatch: got {arr.size}, expected {n_cells}"
                )
            return arr[sort_idx].reshape(nx, ny, nz)

        # --- vector / tensor ---
        if arr.ndim == 2:
            if arr.shape == (3, 1):
                arr = np.repeat(arr, n_cells, axis=1)
            elif arr.shape == (1, 3):
                arr = np.repeat(arr.T, n_cells, axis=1)
            if arr.shape == (n_cells, 3):
                arr = arr.T
            if arr.shape == (3, n_cells):
                return arr[:, sort_idx].reshape(arr.shape[0], nx, ny, nz)

        raise ValueError(
            f"Unsupported field shape {arr.shape}; expected (n_cells,), "
            "(1,), (3, n_cells), (n_cells, 3), (3, 1) or (1, 3)"
        )

    @staticmethod
    def _build_grid_cache(
        X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray
    ) -> Dict:
        """Build mesh cache (lexsort permutation, dimensions, FVM volumes)."""
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)

        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

        x1d = X_raw[sort_idx].reshape(nx, ny, nz)[:, 0, 0]
        y1d = Y_raw[sort_idx].reshape(nx, ny, nz)[0, :, 0]
        z1d = Z_raw[sort_idx].reshape(nx, ny, nz)[0, 0, :]

        dx = np.gradient(x1d)
        dy = np.gradient(y1d)
        dz = np.gradient(z1d)
        volumes = (
            dx[:, np.newaxis, np.newaxis]
            * dy[np.newaxis, :, np.newaxis]
            * dz[np.newaxis, np.newaxis, :]
        )

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis": x1d,
            "y_axis": y1d,
            "z_axis": z1d,
            "volumes": volumes,
        }

    # ══════════════════════════════════════════════════════════════════
    #  Per-time-step processing
    # ══════════════════════════════════════════════════════════════════

    def process_time_step(self, grid: Dict, time_v: float) -> Dict[str, float]:
        """Integrate TKE budget terms at *time_v* → instantaneous power [W].

        Reads all required raw fields once into a cache, then computes
        both the direct OF-derived integrals and the derived (epsilonbt,
        epsilondrag*) terms from that cache.

        Returns dict with keys ``'time'``, each TKE term name (power in W),
        and optionally their non-dimensional counterparts.
        """
        time_dir = self._time_to_dir_name(time_v)
        time_tag = self._time_tag(time_v)
        print(f"Processing t = {time_tag} ...")

        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        sort_idx = grid["sort_idx"]
        volumes = grid["volumes"]

        # ---- 1. read alpha.a (needed for α_f) -------------------------
        integrals: Dict[str, float] = {"time": float(time_v)}
        try:
            alpha_a_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
        except Exception as exc:
            print(f"  ✗ Failed to read alpha.a at t={time_tag}: {exc}")
            return {**integrals,
                    **{k: float("nan") for k in self.output_terms}}
        alpha_a = self._reshape_field(alpha_a_raw, sort_idx, nx, ny, nz)
        alpha_f = 1.0 - alpha_a

        # ---- 2. read all raw fields into cache (once) -----------------
        cache: Dict[str, np.ndarray] = {}

        # scalars
        scalar_list = [
            "magUb", "magUa", "k.b", "omega.b", "TurThird",
            "nut.b", "SUS", "K", "nuFra","pS"
        ]
        for of_name in scalar_list:
            try:
                raw = fluidfoam.readscalar(self.sol, time_dir, of_name)
                cache[of_name] = self._reshape_field(raw, sort_idx, nx, ny, nz)
            except Exception as exc:
                print(f"  ⚠ could not read scalar '{of_name}': {exc}")

        # vectors
        vector_list = ["U.b", "U.a", "grad(alpha.a)"]
        for of_name in vector_list:
            try:
                raw = fluidfoam.readvector(self.sol, time_dir, of_name)
                cache[of_name] = self._reshape_field(raw, sort_idx, nx, ny, nz)
            except Exception as exc:
                print(f"  ⚠ could not read vector '{of_name}': {exc}")


        # ---- 3. epsilonb0 / epsilona0 (OF 中已包含 αf·ρf，直接积分) ----
        if "magUb" in cache:
            ep_f0 = cache["magUb"] 
            integrals["epsilonb0"] = float(np.sum(ep_f0 * volumes))
        else:
            integrals["epsilonb0"] = float("nan")
        if "magUa" in cache:
            ep_s0 = cache["magUa"] 
            integrals["epsilona0"] = float(np.sum(ep_s0 * volumes))
        else:
            integrals["epsilona0"] = float("nan")

        # ---- 4. epsilonbt = 0.09 * k.b * omega.b * alpha_f * rho_f ----
        # k-ω 模型估计的湍流耗散 (Cμ = 0.09)
        if "k.b" in cache and "omega.b" in cache:
            eps_bt = 0.09 * cache["k.b"] * cache["omega.b"] \
                     * alpha_f * self.rho_f
            integrals["epsilonbt"] = float(np.sum(eps_bt * volumes))
        else:
            integrals["epsilonbt"] = float("nan")

        # ----4. epsilonat = 2 * nut.b * alpha_a * rho_s/SUS * TurThird
        if "nut.b" in cache and "SUS" in cache and "TurThird" in cache:
            safe_SUS = cache["SUS"] + 1e-12
            eps_at = 2.0 * cache["nut.b"] * alpha_a * self.rho_s / safe_SUS * cache["TurThird"]
            integrals["epsilonat"] = float(np.sum(eps_at * volumes))
        # ---5. pa = pS * alpha_a * rho_s * g * Ub
        # ---- 5. epsilondrag* terms (particle drag dissipation) --------
        drag_keys = {"K", "U.b", "U.a", "grad(alpha.a)", "SUS",
                     "nut.b", "k.b", "omega.b", "TurThird"}
        if drag_keys.issubset(cache.keys()):
            gamma   = cache["K"]
            ub      = cache["U.b"]            # (3, nx, ny, nz)
            ua      = cache["U.a"]
            grad_a  = cache["grad(alpha.a)"]
            SUS     = cache["SUS"]
            nutb    = cache["nut.b"]
            kb      = cache["k.b"]
            omegab      = cache["omega.b"]
            TurThird     = cache["TurThird"]

            # relative velocity magnitude squared
            udiff_sq = ((ub[0] - ua[0])**2 + (ub[1] - ua[1])**2
                       + (ub[2] - ua[2])**2)
            # dot product of grad(alpha.a) · (Ub - Ua)
            g_dot_du = (grad_a[0] * (ub[0] - ua[0])
                      + grad_a[1] * (ub[1] - ua[1])
                      + grad_a[2] * (ub[2] - ua[2]))

            safe_inv_sus = 1.0 / np.sqrt(SUS + 1e-12)
            safe_omegab      = omegab + 1e-12

            integrals["epsilondrag01"] = float(np.sum(
                gamma * alpha_a * udiff_sq * volumes))
            integrals["epsilondrag02"] = float(np.sum(-1.0 *
                gamma * nutb / (SUS * alpha_f + 1e-12) * g_dot_du * volumes))
            integrals["epsilondragt1"] = float(np.sum(
                2.0 * gamma * alpha_a * (safe_inv_sus - 1.0) * kb * volumes))
            integrals["epsilondragt2"] = float(np.sum(
                gamma * alpha_f * (safe_inv_sus - 1.0) * self.rho_f * kb
                * nutb / (safe_omegab * (SUS + 1e-12)) * TurThird * volumes))
        else:
            for k in ("epsilondrag01", "epsilondrag02",
                      "epsilondragt1", "epsilondragt2"):
                integrals[k] = float("nan")

        # ---- 6. non-dimensionalise ------------------------------------
        if self.nondimensionalize:
            E_ref = 0.42          # J  (reference energy, from energy_terms.py)
            T_ref = self.H_ref / self.U_ref   # s
            P_ref = E_ref / T_ref             # W  (~0.364 W)
            for k in self.output_terms:
                if np.isfinite(integrals.get(k, float("nan"))):
                    integrals[f"{k}*"] = integrals[k] / P_ref

        # ---- 7. print summary -----------------------------------------
        print(f"  alpha_f_sum  = {np.sum(alpha_f * volumes):.8e}  m³")
        for k in self.output_terms:
            v = integrals.get(k, float("nan"))
            if np.isfinite(v):
                print(f"  {k:>14s}  = {v:14.6e}  W")
            else:
                print(f"  {k:>14s}  =  NaN")

        return integrals

    # ══════════════════════════════════════════════════════════════════
    #  I/O & plotting
    # ══════════════════════════════════════════════════════════════════

    def _save_csv(self, df: pd.DataFrame) -> str:
        """Write TKE power + cumulative work to CSV."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(
            self.output_dir, f"TKE_budget_integrals_{self.output_prefix}.csv"
        )
        df.to_csv(csv_path, index=False, float_format="%.8e")
        print(f"\nSaved: {csv_path}")
        return csv_path

    def _plot_terms(self, df: pd.DataFrame) -> str:
        """Plot cumulative TKE work vs time (main) + instantaneous power (inset)."""
        os.makedirs(self.output_dir, exist_ok=True)

        tke_keys = self.output_terms
        active = [
            k for k in tke_keys
            if k in df.columns and np.any(np.isfinite(df[k].to_numpy(dtype=float)))
        ]
        if not active:
            return ""

        time_vals = df["time"].to_numpy(dtype=float)

        # ---- main: cumulative work [J] ---------------------------------------
        cum_keys = [f"cum_{k}" for k in active if f"cum_{k}" in df.columns]
        if cum_keys:
            fig, ax = plt.subplots(figsize=self.fig_size)
            for ck in cum_keys:
                ax.plot(time_vals, df[ck], marker=self.marker, lw=self.lw,
                        label=ck.replace("cum_", ""))
            ax.set_xlabel(r"$t$ (s)", fontsize=self.label_fontsize)
            ax.set_ylabel(r"Cumulative work (J)", fontsize=self.label_fontsize)
            ax.tick_params(axis="both", labelsize=self.tick_fontsize)
            ax.legend(fontsize=self.legend_fontsize)
            ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig_path = os.path.join(
                self.output_dir, f"TKE_budget_integrals_{self.output_prefix}.png"
            )
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            print(f"Saved: {fig_path}")

        # ---- inset: instantaneous power [W] ----------------------------------
        fig2, ax2 = plt.subplots(figsize=self.fig_size)
        for k in active:
            ax2.plot(time_vals, df[k], marker=self.marker, lw=self.lw, label=k)
        ax2.set_xlabel(r"$t$ (s)", fontsize=self.label_fontsize)
        ax2.set_ylabel(r"Power (W)", fontsize=self.label_fontsize)
        ax2.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax2.legend(fontsize=self.legend_fontsize)
        ax2.grid(True, linestyle="--", alpha=0.35)
        fig2.tight_layout()

        power_path = os.path.join(
            self.output_dir, f"TKE_power_{self.output_prefix}.png"
        )
        fig2.savefig(power_path, dpi=300)
        plt.close(fig2)
        print(f"Saved: {power_path}")

        # ---- non-dimensional cumulative --------------------------------------
        if self.nondimensionalize:
            nd_cum = [f"cum_{k}*" for k in active if f"cum_{k}*" in df.columns]
            if nd_cum:
                fig3, ax3 = plt.subplots(figsize=self.fig_size)
                for ck in nd_cum:
                    ax3.plot(time_vals, df[ck], marker=self.marker, lw=self.lw,
                             label=ck.replace("cum_", "").replace("*", "^*"))
                ax3.set_xlabel(r"$t$ (s)", fontsize=self.label_fontsize)
                ax3.set_ylabel(r"$W^*$", fontsize=self.label_fontsize)
                ax3.tick_params(axis="both", labelsize=self.tick_fontsize)
                ax3.legend(fontsize=self.legend_fontsize)
                ax3.grid(True, linestyle="--", alpha=0.35)
                fig3.tight_layout()

                nd_path = os.path.join(
                    self.output_dir,
                    f"TKE_budget_integrals_{self.output_prefix}_nondim.png",
                )
                fig3.savefig(nd_path, dpi=300)
                plt.close(fig3)
                print(f"Saved: {nd_path}")

        return fig_path if cum_keys else power_path

    # ══════════════════════════════════════════════════════════════════
    #  Main entry point
    # ══════════════════════════════════════════════════════════════════

    def run_analysis(self) -> pd.DataFrame:
        """Run TKE budget term integration for all time steps."""
        os.makedirs(self.output_dir, exist_ok=True)

        print("Reading mesh ...")
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        # Try reading cell volumes from 0/V (writeCellVolumes required)
        try:
            volume_raw = fluidfoam.readscalar(self.sol, "0", "V")
            grid["volumes"] = self._reshape_field(
                volume_raw,
                grid["sort_idx"],
                grid["nx"],
                grid["ny"],
                grid["nz"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to read cell-volume field '0/V'. "
                "Generate it with writeCellVolumes before running."
            ) from exc

        print(
            f"  Mesh: {grid['nx']} × {grid['ny']} × {grid['nz']} "
            f"= {grid['nx'] * grid['ny'] * grid['nz']} cells\n"
            f"  Cell-volume sum: {np.sum(grid['volumes']):.9e} m³\n"
        )

        all_results: list = []
        for t in self.times:
            result = self.process_time_step(grid, float(t))
            all_results.append(result)

        df = pd.DataFrame(all_results)
        df = df.sort_values("time").reset_index(drop=True)

        # ---- cumulative time integration (power → energy) --------------------
        print("\nTime-integrating power → cumulative work ...")
        time_vals = df["time"].to_numpy(dtype=float)
        E_ref = 0.42  # J  (same reference as energy_terms.py)
        all_term_keys = self.output_terms
        for k in all_term_keys:
            if k not in df.columns:
                continue
            col = df[k].to_numpy(dtype=float)
            cum = cumulative_trapezoid(col, time_vals, initial=0.0)
            df[f"cum_{k}"] = cum

            if self.nondimensionalize:
                df[f"cum_{k}*"] = cum / E_ref

        print("Done.")

        self._save_csv(df)
        self._plot_terms(df)

        print("\nDone.")
        return df


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = TKEIntegralAnalyzer()
    analyzer.run_analysis()
