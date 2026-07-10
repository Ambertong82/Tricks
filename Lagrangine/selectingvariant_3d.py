import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import numpy as np


@dataclass
class FrontSample:
    time: float
    position: float
    u: np.ndarray
    y: float


class FrontSpeedExtractor:
    def __init__(self):
        # OpenFOAM case directory.
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_5"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/mesh_convergence/case230504_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090428_10"
        

        # Output CSV path.
        self.output_dir = "/home/amber/postpro/selectingvariant_3d"
        # self.output_csv = os.path.join(self.output_dir, "front_speed_2davg_230504_1.csv")

        # Time list to process (0 to 40, step 0.5).
        self.times = np.arange(15, 25, 0.5)
        # self.times = [5,10,15,20,25,30,35,40]

        # Field names and threshold.
        self.alpha_field = "alpha.a"
        self.velocity_field = "U.b"
        self.alpha_threshold = 1e-5
        self.A = 1/2
        self.H = 0.3
        # Additional outputs
        self.output_csv_speed = os.path.join(self.output_dir, "front_speed_2davg_230504_2_matrix.csv")
        self.output_csv_y = os.path.join(self.output_dir, "front_speed_2davg_230504_2_y_matrix.csv")

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _build_grid_cache(
        x_raw: np.ndarray, y_raw: np.ndarray, z_raw: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_axis = np.unique(x_raw)
        y_axis = np.unique(y_raw)
        z_axis = np.unique(z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)

        # Sort by x->y->z to reshape into structured (nx, ny, nz).
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))
        x_3d = x_raw[sort_idx].reshape((nx, ny, nz), order="C")[:, 0, 0]
        y_3d = y_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, :, 0]

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis": x_3d,
            "y_axis": y_3d,
        }

    @staticmethod
    def _reshape_sorted(
        field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int
    ) -> np.ndarray:
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    def _sample_front_at_time(
        self, grid: Dict[str, np.ndarray], time_v: float
    ) -> Optional[FrontSample]:
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        try:
            alpha_raw = fluidfoam.readscalar(self.sol, time_dir, self.alpha_field)
            vel_raw = fluidfoam.readvector(self.sol, time_dir, self.velocity_field)
        except Exception as exc:
            print(f"Read failed at t={time_v}: {exc}")
            return None

        alpha_3d = self._reshape_sorted(alpha_raw, sort_idx, nx, ny, nz)
        vel_3d = self._reshape_sorted(vel_raw, sort_idx, nx, ny, nz)

        # 2D spanwise averages: (nx, ny).
        alpha_2d = np.mean(alpha_3d, axis=2)
        ux_2d = np.mean(vel_3d[0], axis=2)
        uy_2d = np.mean(vel_3d[1], axis=2)

        # Head index: largest x where any y satisfies alpha threshold.
        mask_x = np.any(alpha_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if valid_x.size == 0:
            print(f"No head found at t={time_v} (alpha <= {self.alpha_threshold}).")
            return None

        head_ix = int(valid_x.max())

        # At head x, select y with maximal alpha as representative front point.
        head_row = alpha_2d[head_ix, :]-self.A*self.H
        head_iy = int(np.argmax(head_row))

        head_x = float(grid["x_axis"][head_ix])
        head_y = float(grid["y_axis"][head_iy])

        # target x-coordinate: head - A*H (nearest grid index)
        target_x = head_x - self.A * self.H
        x_axis = grid["x_axis"]
        ix_target = int(np.argmin(np.abs(x_axis - target_x)))

        # ux along entire y at ix_target
        ux_along_y = ux_2d[ix_target, :].astype(float)

        return FrontSample(
            time=float(time_v),
            position=float(x_axis[ix_target]),
            u=ux_along_y,  # now u is an array along y
            y=float(head_y),
        )

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(x_raw, y_raw, z_raw)

        samples = []
        for t in self.times:
            sample = self._sample_front_at_time(grid, float(t))
            if sample is not None:
                samples.append(sample)
                print(f"t={sample.time:.3f}, position={sample.position:.6f}, y={sample.y:.6f}")

        if len(samples) == 0:
            print("No samples collected.")
            return

        # Prepare y axis (columns)
        y_axis = grid["y_axis"]
        y_headers = ",".join([f"{yv:.10g}" for yv in y_axis])

        # Write speed matrix CSV:
        # first row: t, then time values
        # first column: y values
        # cell(i, j): ux at y_i and time_j
        with open(self.output_csv_speed, "w", encoding="utf-8") as fsp:
            fsp.write("t," + ",".join([f"{s.time:.10g}" for s in samples]) + "\n")
            for y_index, y_value in enumerate(y_axis):
                row_values = [f"{y_value:.10g}"]
                for sample in samples:
                    row_values.append(f"{float(np.asarray(sample.u)[y_index]):.10g}")
                fsp.write(",".join(row_values) + "\n")

        # Write y matrix CSV:
        # same layout as the speed matrix, but each cell contains the y coordinate.
        with open(self.output_csv_y, "w", encoding="utf-8") as fy:
            fy.write("t," + ",".join([f"{s.time:.10g}" for s in samples]) + "\n")
            for y_index, y_value in enumerate(y_axis):
                row_values = [f"{y_value:.10g}"]
                for _sample in samples:
                    row_values.append(f"{y_value:.10g}")
                fy.write(",".join(row_values) + "\n")

        print(
            f"Saved matrix CSVs with {len(samples)} times to: "
            f"{self.output_csv_speed} and {self.output_csv_y}"
        )


if __name__ == "__main__":
    extractor = FrontSpeedExtractor()
    extractor.run()