"""
Extract nondimensional streamwise velocity (Ux) and TKE (k.b) vertical
profiles at fixed head-relative positions (0.25H0, 0.5H0, 0.8H0, 1.0H0,
1.2H0, 1.5H0, 2.0H0).

Nondimensionalization:
  y*  = y  / H0          (H0 = 0.3  m)
  Ux* = Ux / U_ref       (U_ref = 0.255 m/s)
  k*  = k  / (0.5U_ref^2)

For 3D cases, fields are spanwise-averaged first; for 2D cases (nz=1)
the same code works without modification.

Framework follows TKE_budget_compare.py and vorticitytransport_ofcal.py.

Output: one CSV per time step — all values are nondimensional (except
the reference x_target columns which give the dimensional sampling
location). y_over_H0 is the first column, followed by columns for
each target position.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import fluidfoam
import numpy as np
import pandas as pd


@dataclass
class TimeStepData:
    """Container for spanwise-averaged fields at one time step."""

    time: float
    x_axis: np.ndarray  # 1D x coordinates
    y_axis: np.ndarray  # 1D y coordinates
    alpha_a: np.ndarray  # (nx, ny) after spanwise averaging
    ubx: np.ndarray  # (nx, ny) streamwise velocity component
    tke: np.ndarray  # (nx, ny) turbulent kinetic energy


class UxTKEProfileExtractor:
    """Extract vertical profiles of Ux and TKE at fixed head-relative positions."""

    def __init__(self):
        # ---------- OpenFOAM case path ----------
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_12"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_2"
        # ---------- Output directory ----------
        self.output_dir = "/home/amber/postpro/Lagrangian/profiles/case230604_2"

        # ---------- Times to process ----------
        self.times = [5, 7,12, 15,20, 35]

        # ---------- Physical parameters ----------
        self.H0 = 0.3  # channel / reference height
        self.U_ref = 0.255  # reference velocity (m/s)
        self.tke_scale = 0.5 * self.U_ref**2  # reference TKE = 0.5 * U_ref^2
        self.target_positions = [0.25, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0,5.0]  # distances behind head (multiples of H0)

        # ---------- Head detection ----------
        self.alpha_threshold = 1e-5

    # ------------------------------------------------------------------
    # Static helpers (same pattern as the reference scripts)
    # ------------------------------------------------------------------

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _build_grid_cache(X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract sorted 1D axes and the lexsort index from flattened OpenFOAM mesh."""
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
        }

    @staticmethod
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        """Reshape flattened OpenFOAM field to structured (nx, ny, nz) or (3, nx, ny, nz)."""
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_fields(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepData]:
        """Read alpha.a, U.b, k.b from OpenFOAM and spanwise-average if nz > 1."""
        print(f"\n>>> Loading t = {time_v}")
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]
        is_3d = nz > 1

        print(f"    Grid shape: nx={nx}, ny={ny}, nz={nz} ({'3D' if is_3d else '2D / quasi-2D'})")

        try:
            alpha_a_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
            ub_raw = fluidfoam.readvector(self.sol, time_dir, "U.b")
            kb_raw = fluidfoam.readscalar(self.sol, time_dir, "k.b")
        except Exception as exc:
            print(f"    Read failed at t={time_v}: {exc}")
            return None

        # Reconstruct structured fields
        alpha_a_3d = self._reshape_sorted(alpha_a_raw, sort_idx, nx, ny, nz)
        ub_3d = self._reshape_sorted(ub_raw, sort_idx, nx, ny, nz)
        kb_3d = self._reshape_sorted(kb_raw, sort_idx, nx, ny, nz)

        # Spanwise average (no-op when nz == 1)
        alpha_a_2d = np.mean(alpha_a_3d, axis=2)
        ubx_2d = np.mean(ub_3d[0], axis=2)  # x-component of velocity
        kb_2d = np.mean(kb_3d, axis=2)

        return TimeStepData(
            time=float(time_v),
            x_axis=grid["x_axis"],
            y_axis=grid["y_axis"],
            alpha_a=alpha_a_2d,
            ubx=ubx_2d,
            tke=kb_2d,
        )

    # ------------------------------------------------------------------
    # Head detection
    # ------------------------------------------------------------------

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        """Find the last x-index where alpha.a exceeds threshold at any y."""
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if len(valid_x) == 0:
            return None
        return int(valid_x.max())

    # ------------------------------------------------------------------
    # Profile extraction at target positions
    # ------------------------------------------------------------------

    def _extract_profiles(self, data: TimeStepData, head_x: float) -> pd.DataFrame:
        """
        For each target distance behind the head * H0,
        find the nearest x-index and extract nondimensional vertical profiles.

        Nondimensionalization:
          y*  = y  / H0
          Ux* = Ux / U_ref
          k*  = k  / (0.5 * U_ref^2)
          alpha_a — already dimensionless.
        """
        y_vals = data.y_axis
        x_vals = data.x_axis

        # Nondimensional y
        profiles: Dict[str, List] = {"y_over_H0": list(y_vals / self.H0)}

        for target in self.target_positions:
            target_x = head_x - target * self.H0
            idx = int(np.argmin(np.abs(x_vals - target_x)))
            actual_x = float(x_vals[idx])

            tag = f"{target:.1f}H0".replace(".", "p")
            profiles[f"x_target_{tag}"] = actual_x  # dimensional, for reference
            profiles[f"Uxstar_{tag}"] = list(data.ubx[idx, :] / self.U_ref)
            profiles[f"kstar_{tag}"] = list(data.tke[idx, :] / self.tke_scale)
            profiles[f"alpha_{tag}"] = list(data.alpha_a[idx, :])

            print(f"    x_target = head_x - {target}*H0 = {target_x:.4f}, "
                  f"nearest x = {actual_x:.4f} (idx={idx})")

        return pd.DataFrame(profiles)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[pd.DataFrame]:
        data = self._load_fields(grid, float(time_v))
        if data is None:
            return None

        head_idx = self._locate_head_index(data.alpha_a)
        if head_idx is None:
            print(f"    No alpha.a > threshold ({self.alpha_threshold}) at t={time_v}. Skip.")
            return None

        head_x = float(data.x_axis[head_idx])
        print(f"    Head: x={head_x:.4f} (idx={head_idx})")

        df = self._extract_profiles(data, head_x)
        df.insert(0, "time", float(time_v))
        return df

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        print("=" * 60)
        print("Ux & TKE Profile Extractor")
        print("=" * 60)

        # Build mesh cache
        print("\nReading mesh ...")
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        print(f"  nx={nx}, ny={ny}, nz={nz}")
        print(f"  x-range: [{grid['x_axis'].min():.4f}, {grid['x_axis'].max():.4f}]")
        print(f"  y-range: [{grid['y_axis'].min():.6f}, {grid['y_axis'].max():.6f}]")

        for t in self.times:
            df = self.process_time_step(grid, float(t))
            if df is None:
                continue

            csv_name = f"Ux_TKE_profiles_t{float(t):g}.csv"
            csv_path = os.path.join(self.output_dir, csv_name)
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

        print("\nDone.")


if __name__ == "__main__":
    extractor = UxTKEProfileExtractor()
    extractor.run_analysis()
