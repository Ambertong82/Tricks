#!/usr/bin/env python3
"""
Global volume-integrated energy terms and particle centre-of-mass height
for turbidity currents.

Computes four domain-integrated quantities at each output time:
  Ep(t)       = ∫Ω (ρs - ρf) αs g y dV                     [potential energy]
  Ekm(t)      = ∫Ω [½ αf ρf |uf|² + ½ αs ρs |us|²] dV      [mean kinetic energy]
  Ekf(t)      = ∫Ω αf ρf kf dV                              [turbulent kinetic energy]
  y_s_mean(t) = ∫Ω (αs · y) dV / ∫Ω (αs) dV                 [particle centre-of-mass height]

Integrands are evaluated on the full 3D structured mesh and summed with the
**FVM cell volumes** reconstructed from the structured grid spacing (central
differences of the cell-centre coordinates --- identical to the control
volumes OpenFOAM uses for a Cartesian mesh).  The file follows the same
fluidfoam + structured-grid convention used elsewhere in this project
(e.g. TKE_budget_ofcal.py, vorticitytransport_ofcal.py).

TKE budget term volume integrals (kprod, kdiss, drag1-3, kdiff1, ksource)
have been moved to the companion script ``tke_integral_terms.py`` in this
directory.
"""

import os
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Optional LaTeX-style rendering (comment out if no TeX available)
# ---------------------------------------------------------------------------
# plt.rc("text", usetex=True)
# plt.rc("font", family="serif")


class EnergyTermsAnalyzer:
    """Compute and output global energy integrals Ep, Ekm, Ekf."""

    # ------------------------------------------------------------------
    # User-configurable parameters
    # ------------------------------------------------------------------
    def __init__(self):
        # OpenFOAM case directory (adjust to your case)
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230704_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090704_2"
        # self.sol = "/media/amber/Elements/Bonnecaze/case090327_12"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2"
        # Output
        self.output_dir = "/home/amber/postpro/energy_terms"
        self.output_prefix = "tc2d_09"

        # Time steps to process  (can also be floats, e.g. [12.5, 15.0])
        self.times = np.arange(0.5, 40, 0.5).tolist()

        # Physical parameters
        self.rho_s = 3217.0       # particle density  [kg / m³]
        self.rho_f = 1000.0       # fluid density     [kg / m³]
        self.g = 9.81             # gravity           [m / s²]

        # Reference scales for non-dimensionalisation
        self.U_ref = 0.26          # bulk velocity     [m / s]
        self.H_ref = 0.3           # reference height  [m]
        self.V_ref = 0.0117

        # When True, also write non-dimensional columns (divided by ρf U² H³)
        self.nondimensionalize = True

        # Plotting style
        self.fig_size = (10, 6)
        self.title_fontsize = 16
        self.label_fontsize = 14
        self.tick_fontsize = 12
        self.legend_fontsize = 12
        self.marker = "o"
        self.lw = 2.0

    # ══════════════════════════════════════════════════════════════════
    #  Static / helper methods (consistent with existing project files)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        """OpenFOAM time-directory name (no trailing zeros)."""
        return f"{float(time_v):g}"

    @staticmethod
    def _time_tag(time_v: float) -> str:
        """Short numeric tag for file names, e.g. '5.00'."""
        return f"{time_v:.2f}"

    @staticmethod
    def _reshape_field(
        field_flat: np.ndarray,
        sort_idx: np.ndarray,
        nx: int,
        ny: int,
        nz: int,
    ) -> np.ndarray:
        """Reshape a flat OpenFOAM field back to the structured (nx, ny, nz)
        or (nComp, nx, ny, nz) layout using the lexsort permutation.

        Mirrors the logic in dimotaski.py / vorticitytransport_ofcal.py.
        """
        n_cells = nx * ny * nz
        arr = np.asarray(field_flat)

        if sort_idx.size != n_cells:
            raise ValueError(
                f"sort_idx size mismatch: got {sort_idx.size}, expected {n_cells}"
            )

        # --- scalar ---
        if arr.ndim == 1:
            # fluidfoam sometimes returns a single value for uniform fields
            if arr.size == 1:
                arr = np.full(n_cells, float(arr[0]), dtype=arr.dtype)
            if arr.size != n_cells:
                raise ValueError(
                    f"scalar field size mismatch: got {arr.size}, expected {n_cells}"
                )
            return arr[sort_idx].reshape(nx, ny, nz)

        # --- vector / tensor ---
        if arr.ndim == 2:
            # uniform vector:   (3, 1) or (1, 3)
            if arr.shape == (3, 1):
                arr = np.repeat(arr, n_cells, axis=1)
            elif arr.shape == (1, 3):
                arr = np.repeat(arr.T, n_cells, axis=1)

            # fluidfoam may return (n_cells, 3) or (3, n_cells) — normalise
            if arr.shape == (n_cells, 3):
                arr = arr.T
            if arr.shape == (3, n_cells):
                return arr[:, sort_idx].reshape(arr.shape[0], nx, ny, nz)

        raise ValueError(
            f"Unsupported field shape {arr.shape}; expected (n_cells,), "
            "(1,), (3, n_cells), (n_cells, 3), (3, 1) or (1, 3)"
        )

    @staticmethod
    def _safe_gradient(coords: np.ndarray) -> np.ndarray:
        """Cell spacing from 1-D cell-centre coordinates.

        np.gradient needs at least 2 points, so for a singleton axis (e.g. the
        spanwise direction of a 2D case, nz=1) it cannot be used.  Fall back to
        a placeholder spacing of 1.0; this is only a fallback, since
        run_analysis() overwrites ``volumes`` with the real OpenFOAM
        cell-volume field '0/V' afterwards.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.size < 2:
            return np.ones_like(coords)
        return np.gradient(coords)

    @staticmethod
    def _build_grid_cache(
        X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray
    ) -> Dict:
        """Build the mesh cache used to reorder OpenFOAM cell fields.

        Returns lexsort permutation, grid dimensions, 1-D coordinate arrays,
        and FVM cell volumes estimated from the structured grid spacing.
        """
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)

        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

        # 1-D coordinate arrays (cell-centre locations)
        x1d = X_raw[sort_idx].reshape(nx, ny, nz)[:, 0, 0]
        y1d = Y_raw[sort_idx].reshape(nx, ny, nz)[0, :, 0]
        z1d = Z_raw[sort_idx].reshape(nx, ny, nz)[0, 0, :]

        # Cell volumes from structured grid spacing
        # (z-spacing falls back to 1.0 for 2D cases with a single spanwise cell)
        dx = EnergyTermsAnalyzer._safe_gradient(x1d)
        dy = EnergyTermsAnalyzer._safe_gradient(y1d)
        dz = EnergyTermsAnalyzer._safe_gradient(z1d)
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

    def process_time_step(
        self, grid: Dict, time_v: float
    ) -> Optional[Dict[str, float]]:
        """Read fields at *time_v* and return the three energy integrals.

        Parameters
        ----------
        grid : dict
            Output of :meth:`_build_grid_cache`.
        time_v : float
            Physical time to process.

        Returns
        -------
        result : dict or None
            Keys ``'time'``, ``'Ep'``, ``'Ekm'``, ``'Ekf'``, plus optionally
            their non-dimensional counterparts.  ``None`` when required fields
            cannot be read.
        """
        time_dir = self._time_to_dir_name(time_v)
        time_tag = self._time_tag(time_v)
        print(f"Processing t = {time_tag} ...")

        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        sort_idx = grid["sort_idx"]
        y1d = grid["y_axis"]
        volumes = grid["volumes"]          # (nx, ny, nz)

        # ---- read required fields -------------------------------------------
        try:
            alpha_a_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
            ub_raw = fluidfoam.readvector(self.sol, time_dir, "U.b")
        except Exception as exc:
            print(f"  ✗ Failed to read alpha.a / U.b at t={time_tag}: {exc}")
            return None

        alpha_a = self._reshape_field(alpha_a_raw, sort_idx, nx, ny, nz)
        ub = self._reshape_field(ub_raw, sort_idx, nx, ny, nz)       # (3, nx, ny, nz)

        # Volume fractions
        alpha_s = alpha_a                     # solid (particle) fraction
        alpha_f = 1.0 - alpha_a               # fluid fraction

        # ---- Ⅰ. Mean kinetic energy (Ekm) -----------------------------------
        # Fluid contribution:  ½ αf ρf |uf|²
        uf_mag2 = ub[0] ** 2 + ub[1] ** 2 + ub[2] ** 2
        integrand_ekm_f = 0.5 * alpha_f * self.rho_f * uf_mag2
        ekm_fluid = float(np.sum(integrand_ekm_f * volumes))

        # Particle contribution:  ½ αs ρs |us|²   (optional, U.a may be missing)
        ekm_particle = 0.0
        try:
            ua_raw = fluidfoam.readvector(self.sol, time_dir, "U.a")
            ua = self._reshape_field(ua_raw, sort_idx, nx, ny, nz)
            us_mag2 = ua[0] ** 2 + ua[1] ** 2 + ua[2] ** 2
            integrand_ekm_s = 0.5 * alpha_s * self.rho_s * us_mag2
            ekm_particle = float(np.sum(integrand_ekm_s * volumes))
        except Exception:
            print(f"  (U.a not available; particle kinetic energy = 0)")

        ekm = ekm_fluid + ekm_particle

        # ---- Ⅱ. Potential energy (Ep) ---------------------------------------
        # (ρs - ρf) αs g y
        # y for each cell is y1d[j]; broadcast to (nx, ny, nz)
        integrand_ep = (
           (self.rho_s - self.rho_f) * alpha_s * self.g * y1d[np.newaxis, :, np.newaxis]
        )
        ep = float(np.sum(integrand_ep * volumes))

        # ---- Ⅲ. Turbulent kinetic energy (Ekf) ------------------------------
        # αf ρf kf
        try:
            kf_raw = fluidfoam.readscalar(self.sol, time_dir, "k.b")
            # sus = fluidfoam.readscalar(self.sol, time_dir, "SUS")
            kf = self._reshape_field(kf_raw, sort_idx, nx, ny, nz)
            # sus = self._reshape_field(sus, sort_idx, nx, ny, nz )
            integrand_ekf = alpha_f * self.rho_f * kf
            ekf = float(np.sum(integrand_ekf * volumes))
            integrand_eks = alpha_s * self.rho_s * kf 
            eks = float(np.sum(integrand_eks * volumes))
        except Exception as exc:
            print(f"  ✗ Failed to read k.b at t={time_tag}: {exc}")
            return None
        


        except Exception:
            print(f"  (k.a not available; particle turbulent kinetic energy = 0)")
            eks = 0.0

        # ---- Ⅴ. Particle centre-of-mass height (y_s_mean) --------------------
        # y_s_mean = ∫(αs · y) dV / ∫(αs) dV
        integrand_ys_num = alpha_s * y1d[np.newaxis, :, np.newaxis]  # αs · y
        integrand_ys_den = alpha_s                                  # αs
        ys_num = float(np.sum(integrand_ys_num * volumes))
        ys_den = float(np.sum(integrand_ys_den * volumes))
        ys_mean = ys_num / ys_den if abs(ys_den) > 1e-20 else 0.0

        # ---- assemble result ------------------------------------------------
        result = {
            "time": float(time_v),
            "Ep": ep,
            "Ekm": ekm,
            "Ekf": ekf,
            "Eks": eks,
            "y_s_mean": ys_mean,
        }

        # Non-dimensionalise  (reference energy = ρf * U_ref² * H_ref³)
        if self.nondimensionalize:
            E_ref = 0.42 #  E_ref = ((self.rho_s - self.rho_f)* 0.011* self.g* 0.15* 0.3* 0.26* (0.3 / 2))

            result["Ep*"] = ep / E_ref
            result["Ekm*"] = ekm / E_ref
            result["Ekf*"] = ekf / E_ref
            result["Eks*"] = eks / E_ref

        # Print summary
        print(f"  Ep  = {ep:14.6e}  J")
        print(f"  Ekm = {ekm:14.6e}  J")
        print(f"  Ekf = {ekf:14.6e}  J")
        print(f"  Eks = {eks:14.6e}  J")
        print(f"  y_s_mean = {ys_mean:.6e}  m")
        if self.nondimensionalize:
            print(f"  Ep*  = {result['Ep*']:10.6f}   (ref. energy = {E_ref:.3e} J)")
            print(f"  Ekm* = {result['Ekm*']:10.6f}")
            print(f"  Ekf* = {result['Ekf*']:10.6f}")
            print(f"  Eks* = {result['Eks*']:10.6f}")

        return result

    # ══════════════════════════════════════════════════════════════════
    #  I/O helpers
    # ══════════════════════════════════════════════════════════════════

    def _save_csv(self, results: pd.DataFrame) -> str:
        """Write the aggregated energy results to CSV."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(
            self.output_dir,
            f"energy_terms_{self.output_prefix}.csv",
        )
        results.to_csv(csv_path, index=False, float_format="%.8e")
        print(f"\nSaved: {csv_path}")
        return csv_path

    # ══════════════════════════════════════════════════════════════════
    #  Plotting
    # ══════════════════════════════════════════════════════════════════

    def _plot_energies(self, results: pd.DataFrame) -> str:
        """Plot dimensional (and optionally non-dimensional) energies vs time."""
        os.makedirs(self.output_dir, exist_ok=True)

        time_vals = results["time"].to_numpy(dtype=float)

        # ---- dimensional plot -----------------------------------------------
        fig, ax = plt.subplots(figsize=self.fig_size)

        ax.plot(time_vals, results["Ep"], marker=self.marker, lw=self.lw,
                label=r"$E_p$")
        ax.plot(time_vals, results["Ekm"], marker=self.marker, lw=self.lw,
                label=r"$E_{km}$")
        ax.plot(time_vals, results["Ekf"], marker=self.marker, lw=self.lw,
                label=r"$E_{kf}$")
        ax.plot(time_vals, results["Eks"], marker=self.marker, lw=self.lw,
                label=r"$E_{ks}$")

        ax.set_xlabel(r"$t$ (s)", fontsize=self.label_fontsize)
        ax.set_ylabel(r"Energy (J)", fontsize=self.label_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax.legend(fontsize=self.legend_fontsize)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()

        dim_path = os.path.join(self.output_dir, f"energy_terms_{self.output_prefix}.png")
        fig.savefig(dim_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {dim_path}")

        # ---- centre-of-mass height plot -------------------------------------
        if "y_s_mean" in results.columns:
            fig3, ax3 = plt.subplots(figsize=self.fig_size)

            ax3.plot(time_vals, results["y_s_mean"], marker=self.marker,
                     lw=self.lw, color="C3", label=r"$y_s^{mean}$")

            ax3.set_xlabel(r"$t$ (s)", fontsize=self.label_fontsize)
            ax3.set_ylabel(r"$y_s^{mean}$ (m)", fontsize=self.label_fontsize)
            ax3.tick_params(axis="both", labelsize=self.tick_fontsize)
            ax3.legend(fontsize=self.legend_fontsize)
            ax3.grid(True, linestyle="--", alpha=0.35)
            fig3.tight_layout()

            ys_path = os.path.join(
                self.output_dir, f"y_s_mean_{self.output_prefix}.png"
            )
            fig3.savefig(ys_path, dpi=300)
            plt.close(fig3)
            print(f"Saved: {ys_path}")

        # ---- non-dimensional plot (if available) ----------------------------
        if self.nondimensionalize and "Ep*" in results.columns:
            fig2, ax2 = plt.subplots(figsize=self.fig_size)

            ax2.plot(time_vals, results["Ep*"], marker=self.marker, lw=self.lw,
                     label=r"$E_p^*$")
            ax2.plot(time_vals, results["Ekm*"], marker=self.marker, lw=self.lw,
                     label=r"$E_{km}^*$")
            ax2.plot(time_vals, results["Ekf*"], marker=self.marker, lw=self.lw,
                     label=r"$E_{kf}^*$")
            ax2.plot(time_vals, results["Eks*"], marker=self.marker, lw=self.lw,
                     label=r"$E_{ks}^*$")

            ax2.set_xlabel(r"$t$ (s)", fontsize=self.label_fontsize)
            ax2.set_ylabel(r"$E^* = E / (\rho_f U^2 H^3)$",
                           fontsize=self.label_fontsize)
            ax2.tick_params(axis="both", labelsize=self.tick_fontsize)
            ax2.legend(fontsize=self.legend_fontsize)
            ax2.grid(True, linestyle="--", alpha=0.35)
            fig2.tight_layout()

            nd_path = os.path.join(
                self.output_dir, f"energy_terms_{self.output_prefix}_nondim.png"
            )
            fig2.savefig(nd_path, dpi=300)
            plt.close(fig2)
            print(f"Saved: {nd_path}")

        return dim_path

    # ══════════════════════════════════════════════════════════════════
    #  Main entry point
    # ══════════════════════════════════════════════════════════════════

    def run_analysis(self) -> pd.DataFrame:
        """Run the analysis for all time steps and save outputs."""
        os.makedirs(self.output_dir, exist_ok=True)

        print("Reading mesh ...")
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

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
                "Failed to read OpenFOAM cell-volume field '0/V'. "
                "Generate it with writeCellVolumes before running this script."
            ) from exc

        print(
            f"  Mesh: {grid['nx']} × {grid['ny']} × {grid['nz']} "
            f"= {grid['nx'] * grid['ny'] * grid['nz']} cells\n"
            f"  Cell-volume sum: {np.sum(grid['volumes']):.9e} m^3\n"
        )

        all_results: list = []
        for t in self.times:
            result = self.process_time_step(grid, float(t))
            if result is not None:
                all_results.append(result)

        if not all_results:
            print("No valid data processed.  Exiting.")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df = df.sort_values("time").reset_index(drop=True)

        self._save_csv(df)
        self._plot_energies(df)

        print("\nDone.")
        return df


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = EnergyTermsAnalyzer()
    analyzer.run_analysis()
