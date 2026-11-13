"""Extract the turbidity-current front line across the spanwise (z) direction.

For every time step and every spanwise grid point z, the front/head x position
is the largest x grid coordinate where the spanwise slice of alpha.a exceeds
the threshold (alpha_threshold = 1e-5).  This yields one front line
head_x(z) per time step; the spanwise average of that line is the mean head
position at that time.

Outputs (one CSV per time step + one combined summary):
  front_line_alpha1e-5_t<time>.csv   -- the line: z, head_x, head_y, alpha_head
  front_summary_alpha1e-5.csv        -- spanwise-averaged head per time step
"""

import os
from typing import Dict, Optional

import fluidfoam
import numpy as np
import pandas as pd


class FrontLineSpanwise:
    def __init__(self):
        # OpenFOAM case (same case as the vorticity post-processing script).
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_12"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"

        # Time directories present in the case: 0, 5, 10, 15, 20, 25, 30, 35, 40.
        self.times = np.arange(1, 40, 0.5)

        # Output CSV directory.
        self.output_dir = "/home/amber/postpro/frontposition_turbidity23"

        # Field and threshold.
        self.alpha_field = "alpha.a"
        self.alpha_threshold = 1e-5

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    def _build_sorted_mesh(self) -> Dict[str, np.ndarray]:
        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        nx, ny, nz = (
            len(np.unique(x_raw)),
            len(np.unique(y_raw)),
            len(np.unique(z_raw)),
        )
        # Sort by x -> y -> z so each field reshapes to (nx, ny, nz).
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))

        x_3d = x_raw[sort_idx].reshape(nx, ny, nz)
        y_3d = y_raw[sort_idx].reshape(nx, ny, nz)
        z_3d = z_raw[sort_idx].reshape(nx, ny, nz)

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis": x_3d[:, 0, 0],  # x of every cell column (nx,)
            "y_axis": y_3d[0, :, 0],  # y of every cell column (ny,)
            "z_axis": z_3d[0, 0, :],  # z of every spanwise plane (nz,)
        }

    def _front_line_at_time(
        self, grid: Dict[str, np.ndarray], time_v: float
    ) -> Optional[pd.DataFrame]:
        """Front head_x(z) for one time step, plus per-z head metadata."""
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        try:
            alpha_raw = fluidfoam.readscalar(self.sol, time_dir, self.alpha_field)
        except Exception as exc:
            print(f"Read failed at t={time_v}: {exc}")
            return None

        alpha_3d = alpha_raw[sort_idx].reshape(nx, ny, nz)

        z_axis = grid["z_axis"]
        x_axis = grid["x_axis"]
        y_axis = grid["y_axis"]

        head_x = np.full(nz, np.nan, dtype=float)
        head_y = np.full(nz, np.nan, dtype=float)
        alpha_head = np.full(nz, np.nan, dtype=float)

        for iz in range(nz):
            alpha_xy = alpha_3d[:, :, iz]  # (nx, ny) slice at this z
            mask_x = np.any(alpha_xy > self.alpha_threshold, axis=1)
            valid_x = np.where(mask_x)[0]
            if valid_x.size == 0:
                continue

            head_ix = int(valid_x.max())
            head_x[iz] = float(x_axis[head_ix])

            # At the head column, pick the y with maximal alpha as a reference.
            head_iy = int(np.argmax(alpha_xy[head_ix, :]))
            head_y[iz] = float(y_axis[head_iy])
            alpha_head[iz] = float(alpha_xy[head_ix, head_iy])

        return pd.DataFrame(
            {"z": z_axis, "head_x": head_x, "head_y": head_y, "alpha_head": alpha_head}
        )

    def run_analysis(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        grid = self._build_sorted_mesh()

        summary_rows = []
        for t in self.times:
            time_dir = self._time_to_dir_name(float(t))
            line = self._front_line_at_time(grid, float(t))
            if line is None:
                continue

            head_x = line["head_x"].to_numpy(dtype=float)
            n_valid = int(np.isfinite(head_x).sum())
            if n_valid == 0:
                print(f"t={time_dir}: no head found (alpha <= {self.alpha_threshold:g})")
                continue

            mean_head = float(np.nanmean(head_x))
            std_head = float(np.nanstd(head_x))
            print(
                f"t={time_dir}: head_x(z) in [{np.nanmin(head_x):.4f}, "
                f"{np.nanmax(head_x):.4f}], spanwise mean={mean_head:.4f} "
                f"(std={std_head:.4f}, n={n_valid}/{head_x.size})"
            )

            # One CSV per time step: the front line.
            line_path = os.path.join(
                self.output_dir,
                f"front_line_alpha{self.alpha_threshold:g}_t{time_dir}.csv",
            )
            line.to_csv(line_path, index=False)
            print(f"  saved: {line_path}")

            summary_rows.append(
                {
                    "time": float(t),
                    "head_x_mean": mean_head,
                    "head_x_std": std_head,
                    "head_x_min": float(np.nanmin(head_x)),
                    "head_x_max": float(np.nanmax(head_x)),
                    "n_valid": n_valid,
                }
            )

        # Combined summary: spanwise-averaged head position vs time.
        if summary_rows:
            summary_path = os.path.join(
                self.output_dir, f"front_summary_alpha{self.alpha_threshold:g}.csv"
            )
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(f"  saved: {summary_path}")


if __name__ == "__main__":
    FrontLineSpanwise().run_analysis()
