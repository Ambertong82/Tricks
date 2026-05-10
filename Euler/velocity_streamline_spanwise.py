import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TimeStepVelocity3D:
    """Container for one time-step fields in structured layout (nx, ny, nz)."""

    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_a: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray


class VelocityStreamlineAnalyzer:
    def __init__(self):
        # OpenFOAM case directory and output directory.
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230311_1"
        # self.output_dir = "/home/amber/postpro/velocity_streamline/tc3d_d23_0327_1"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_11"
        self.output_dir = "/home/amber/postpro/velocity_streamline/tc3d_d09_0327_1"
        self.times = [15,25]

        # alpha.a threshold for current head detection.
        self.alpha_threshold = 1e-5
        # Characteristic height for non-dimensional x.
        self.H = 0.3
        # Velocity field name in OpenFOAM.
        self.velocity_field = "U.b"

        # Plot style.
        self.fig_size = (15, 3)
        self.stream_density = 1.5
        self.stream_linewidth = 1.2
        self.stream_arrowsize = 1.2
        self.speed_percentile = (1.0, 99.0)
        self.dx_stream = 0.004
        self.dy_stream = 0.001
        self.cmap = "coolwarm"
        # Font sizes (customize these to change plot fonts)
        self.title_fontsize = 12
        self.label_fontsize = 12
        self.tick_fontsize = 10
        self.cbar_labelsize = 10
        self.cbar_ticksize = 10

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _build_grid_cache(
        X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray
    ) -> Dict[str, np.ndarray]:
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

        x3d = X_raw[sort_idx].reshape((nx, ny, nz), order="C")[:, 0, 0]
        y3d = Y_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, :, 0]
        z3d = Z_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, 0, :]

        return {
            "sort_idx": sort_idx,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "x_axis_3d": x3d,
            "y_axis_3d": y3d,
            "z_axis_3d": z3d,
        }

    @staticmethod
    def _reshape_sorted(
        field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int
    ) -> np.ndarray:
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if valid_x.size == 0:
            return None
        return int(valid_x.max())

    @staticmethod
    def _write_structured_grid_vtk(
        out_path: str,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        scalar_name: str,
        scalar_field: np.ndarray,
        vector_name: Optional[str] = None,
        vector_x: Optional[np.ndarray] = None,
        vector_y: Optional[np.ndarray] = None,
    ) -> None:
        """Write a 2D legacy VTK STRUCTURED_GRID file.

        The grid is stored in the x-y plane with z=0. Point data can contain one
        scalar field and optionally one 2D vector field.
        """
        nx, ny = scalar_field.shape
        z_2d = np.zeros_like(scalar_field)

        with open(out_path, "w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Spanwise averaged streamline field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"POINTS {nx * ny} float\n")

            for j in range(ny):
                for i in range(nx):
                    f.write(
                        f"{float(x_2d[i, j]):.9e} {float(y_2d[i, j]):.9e} {float(z_2d[i, j]):.9e}\n"
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

            if vector_name is not None and vector_x is not None and vector_y is not None:
                f.write(f"VECTORS {vector_name} float\n")
                for j in range(ny):
                    for i in range(nx):
                        ux = float(vector_x[i, j])
                        uy = float(vector_y[i, j])
                        if not np.isfinite(ux):
                            ux = 0.0
                        if not np.isfinite(uy):
                            uy = 0.0
                        f.write(f"{ux:.9e} {uy:.9e} 0.000000000e+00\n")

    def _load_velocity_3d(
        self, grid: Dict[str, np.ndarray], time_v: float
    ) -> Optional[TimeStepVelocity3D]:
        print(f"\n>>> Processing time: {time_v}")
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        try:
            alpha_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
            alpha_a = self._reshape_sorted(alpha_raw, sort_idx, nx, ny, nz)
        except Exception as exc:
            print(f"Read failed for alpha.a at t={time_v}: {exc}")
            return None

        try:
            vel_raw = fluidfoam.readvector(self.sol, time_dir, self.velocity_field)
            vel_3d = self._reshape_sorted(vel_raw, sort_idx, nx, ny, nz)
            ux, uy, uz = vel_3d[0], vel_3d[1], vel_3d[2]
        except Exception as exc:
            print(f"Read failed for {self.velocity_field} at t={time_v}: {exc}")
            return None

        return TimeStepVelocity3D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            alpha_a=alpha_a,
            ux=ux,
            uy=uy,
            uz=uz,
        )

    def _save_streamline(
        self,
        time_v: float,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        ux_2d: np.ndarray,
        uy_2d: np.ndarray,
        alpha_2d: np.ndarray,
        head_idx: int,
        head_x: float,
    ) -> None:
        out_dir = os.path.join(self.output_dir, f"streamline_t{time_v:.2f}")
        os.makedirs(out_dir, exist_ok=True)

        x_seg = x_axis[: head_idx + 1]
        x_dime = (head_x - x_seg) / self.H
        y_vals = y_axis / self.H

        u_seg = -ux_2d[: head_idx + 1, :]
        v_seg = uy_2d[: head_idx + 1, :]
        a_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)

        # streamplot requires x/y coordinates to be strictly increasing.
        x_order = np.argsort(x_dime)
        x_plot = x_dime[x_order]
        u_plot = u_seg[x_order, :]
        v_plot = v_seg[x_order, :]
        a_plot = a_seg[x_order, :]

        if x_plot.size < 2 or np.any(np.diff(x_plot) <= 0):
            print(f"Non-monotonic x grid at t={time_v}. Skip streamline output.")
            return

        if y_vals.size < 2 or np.any(np.diff(y_vals) <= 0):
            print(f"Non-monotonic y grid at t={time_v}. Skip streamline output.")
            return

        # streamplot prefers equally spaced x and y coordinates.

        x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

        x_stream = np.arange(x_min, x_max + 0.5 * self.dx_stream, self.dx_stream)
        y_stream = np.arange(y_min, y_max + 0.5 * self.dy_stream, self.dy_stream)

        # 1) Interpolate each x-row from original y grid to y_stream.
        u_y = np.full((x_plot.size, y_stream.size), np.nan, dtype=float)
        v_y = np.full((x_plot.size, y_stream.size), np.nan, dtype=float)
        a_y = np.full((x_plot.size, y_stream.size), np.nan, dtype=float)
        for i in range(x_plot.size):
            row_u = u_plot[i, :]
            row_v = v_plot[i, :]
            row_a = a_plot[i, :]
            mask_u = np.isfinite(row_u)
            mask_v = np.isfinite(row_v)
            mask_a = np.isfinite(row_a)
            if np.count_nonzero(mask_u) >= 2:
                u_y[i, :] = np.interp(
                    y_stream, y_vals[mask_u], row_u[mask_u], left=np.nan, right=np.nan
                )
            if np.count_nonzero(mask_v) >= 2:
                v_y[i, :] = np.interp(
                    y_stream, y_vals[mask_v], row_v[mask_v], left=np.nan, right=np.nan
                )
            if np.count_nonzero(mask_a) >= 2:
                a_y[i, :] = np.interp(
                    y_stream, y_vals[mask_a], row_a[mask_a], left=np.nan, right=np.nan
                )

        # 2) Interpolate each y-column from original x grid to x_stream.
        u_stream = np.full((x_stream.size, y_stream.size), np.nan, dtype=float)
        v_stream = np.full((x_stream.size, y_stream.size), np.nan, dtype=float)
        a_stream = np.full((x_stream.size, y_stream.size), np.nan, dtype=float)
        for j in range(y_stream.size):
            col_u = u_y[:, j]
            col_v = v_y[:, j]
            col_a = a_y[:, j]
            mask_u = np.isfinite(col_u)
            mask_v = np.isfinite(col_v)
            mask_a = np.isfinite(col_a)
            if np.count_nonzero(mask_u) >= 2:
                u_stream[:, j] = np.interp(
                    x_stream, x_plot[mask_u], col_u[mask_u], left=np.nan, right=np.nan
                )
            if np.count_nonzero(mask_v) >= 2:
                v_stream[:, j] = np.interp(
                    x_stream, x_plot[mask_v], col_v[mask_v], left=np.nan, right=np.nan
                )
            if np.count_nonzero(mask_a) >= 2:
                a_stream[:, j] = np.interp(
                    x_stream, x_plot[mask_a], col_a[mask_a], left=np.nan, right=np.nan
                )

        # Background cloud: alpha.a (coolwarm)
        xx, yy = np.meshgrid(x_plot, y_vals, indexing="ij")
        valid = a_plot[np.isfinite(a_plot)]
        if valid.size == 0:
            print(f"Invalid alpha field at t={time_v}. Skip plotting.")
            return

        if float(np.nanmax(valid)) <= 0.0:
            print(f"Degenerated alpha range at t={time_v}. Skip plotting.")
            return

        alpha_plot = np.clip(a_plot, 0.0, 0.01)

        fig, ax = plt.subplots(figsize=self.fig_size)
        cf = ax.contourf(
            xx,
            yy,
            alpha_plot,
            levels=np.linspace(0, 0.01, 121),
            cmap="coolwarm",
            extend="neither",
        )
        cbar = fig.colorbar(cf, ax=ax, ticks=[0.0, 0.0025, 0.005, 0.0075, 0.01])
        cbar.set_label("alpha.a", fontsize=self.cbar_labelsize)
        cbar.ax.tick_params(labelsize=self.cbar_ticksize)
        cbar.set_ticklabels(["0", "0.0025", "0.005","0.0075","0.01"])

        # Streamlines with arrows
        ax.streamplot(
            x_stream,
            y_stream,
            np.ma.masked_invalid(u_stream.T),
            np.ma.masked_invalid(v_stream.T),
            density=self.stream_density,
            color="gray",
            linewidth=self.stream_linewidth,
            arrowsize=self.stream_arrowsize,
        )

        # alpha = 0.01 contour line
        # alpha_valid = a_plot[np.isfinite(a_plot)]
        # if alpha_valid.size > 0:
        #     a_min = float(np.nanmin(alpha_valid))
        #     a_max = float(np.nanmax(alpha_valid))
        #     if a_min <= 0.01 <= a_max:
        #         ax.contour(
        #             xx,
        #             yy,
        #             a_plot,
        #             levels=[0.01],
        #             colors="w",
        #             linestyles="--",
        #             linewidths=1.2,
        #             zorder=5,
        #         )

        ax.set_title(f"Spanwise-Averaged Streamlines at t={time_v:.2f}s", fontsize=self.title_fontsize)
        ax.set_xlabel(r"$(x_f-x)/H$", fontsize=self.label_fontsize)
        ax.set_ylabel("y/H", fontsize=self.label_fontsize)
        ax.set_xlim(float(np.max(x_plot)), 0.0)
        ax.set_ylim(float(np.min(y_vals)), float(np.max(y_vals)))
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        fig.tight_layout()

        out_png = os.path.join(out_dir, f"streamline_spanwise_t{time_v:.2f}.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved Figure: {out_png}")

        # Save the same streamline field as a legacy VTK structured grid for ParaView.
        x_mesh, y_mesh = np.meshgrid(x_stream, y_stream, indexing="ij")
        out_vtk = os.path.join(out_dir, f"streamline_spanwise_t{time_v:.2f}.vtk")
        self._write_structured_grid_vtk(
            out_vtk,
            x_mesh,
            y_mesh,
            "alpha_a",
            a_stream,
            vector_name="U_spanwise",
            vector_x=u_stream,
            vector_y=v_stream,
        )
        print(f"Saved VTK: {out_vtk}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> None:
        data_3d = self._load_velocity_3d(grid, float(time_v))
        if data_3d is None:
            return

        alpha_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_2d)
        if head_idx is None:
            print(
                f"No alpha.a > threshold ({self.alpha_threshold}) at t={time_v}. Skip output."
            )
            return

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx})")

        ux_2d = np.mean(data_3d.ux, axis=2)
        uy_2d = np.mean(data_3d.uy, axis=2)

        self._save_streamline(
            float(time_v),
            data_3d.x_axis,
            data_3d.y_axis,
            ux_2d,
            uy_2d,
            alpha_2d,
            head_idx,
            head_x,
        )

    def run_analysis(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        for t in self.times:
            self.process_time_step(grid, float(t))


if __name__ == "__main__":
    analyzer = VelocityStreamlineAnalyzer()
    analyzer.run_analysis()
