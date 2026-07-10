import os
from typing import Dict

import fluidfoam
import numpy as np


class AlphaSpanwiseVTKExporter:
    def __init__(self):
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_12"
        # self.output_dir = "/home/amber/postpro/spanwise_vtk/alpha_a/tc3d_d09_0327_12"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        self.output_dir = "/home/amber/postpro/spanwise_vtk/alpha_a/tc3d_d23_0428_4"
        self.times = [ 5, 7, 15, 25, 35]
        self.field_name = "alpha.a"
        self.output_field_name = "alpha_a_spanwise"

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
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))

        x3d = x_raw[sort_idx].reshape((nx, ny, nz), order="C")[:, 0, 0]
        y3d = y_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, :, 0]
        z3d = z_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, 0, :]

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis": x3d,
            "y_axis": y3d,
            "z_axis": z3d,
        }

    @staticmethod
    def _reshape_sorted(
        field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int
    ) -> np.ndarray:
        return field[sort_idx].reshape((nx, ny, nz), order="C")

    @staticmethod
    def _write_structured_grid_vtk(
        out_path: str,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        scalar_name: str,
        scalar_field: np.ndarray,
    ) -> None:
        nx, ny = scalar_field.shape

        with open(out_path, "w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Spanwise averaged alpha field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"POINTS {nx * ny} float\n")

            for j in range(ny):
                for i in range(nx):
                    f.write(
                        f"{float(x_axis[i]):.9e} {float(y_axis[j]):.9e} 0.000000000e+00\n"
                    )

            f.write(f"POINT_DATA {nx * ny}\n")
            f.write(f"SCALARS {scalar_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    val = float(scalar_field[i, j])
                    if not np.isfinite(val):
                        val = -9999.0
                    f.write(f"{val:.9e}\n")

    def export_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> None:
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        print(f"\n>>> Processing t={time_v}")
        alpha_raw = fluidfoam.readscalar(self.sol, time_dir, self.field_name)
        alpha_3d = self._reshape_sorted(alpha_raw, sort_idx, nx, ny, nz)
        alpha_2d = np.mean(alpha_3d, axis=2)

        out_path = os.path.join(
            self.output_dir, f"{self.output_field_name}_t{float(time_v):.2f}.vtk"
        )
        self._write_structured_grid_vtk(
            out_path,
            grid["x_axis"],
            grid["y_axis"],
            self.output_field_name,
            alpha_2d,
        )
        print(f"Saved VTK: {out_path}")

    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(x_raw, y_raw, z_raw)

        for time_v in self.times:
            self.export_time_step(grid, float(time_v))


if __name__ == "__main__":
    AlphaSpanwiseVTKExporter().run()
