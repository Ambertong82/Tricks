"""Three-dimensional vorticity-stretching post-processing.

Coordinate convention (OpenFOAM): x = streamwise, y = vertical,
z = spanwise.  The script averages every vector component along z (axis 2),
then averages the resulting x-y field vertically along y (axis 1).  No
interface reconstruction or interpolation is performed.

Set ``stretching_fields`` to the vector field(s) written by your solver for
the stretching contribution.  The supplied Vort_Advection entries are only
placeholders based on the fields used by vorticitytransport_ofcal.py; replace
them if your stretching term has a different OpenFOAM field name.
"""

import os
from typing import Dict

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


class VorticityStretching3DAnalyzer:
    def __init__(self):
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230327_1"
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_11"
        self.times = [15, 25, 35]
        self.output_root = "/home/amber/postpro/u_vorticity"
        self.output_prefix = "tc3d_09_stretching_3d"

        # Replace these names with the stretching field(s) actually written
        # by the OpenFOAM solver.  Each field must be a volVectorField.
        self.stretching_fields: Dict[str, str] = {
            "adv1": "Vort_Advection1",
            "adv2": "Vort_Advection2",
            "adv3": "Vort_Advection3",
            "adv4": "Vort_Advection4",
            "adv5": "Vort_Advection5",
        }

        self.alpha_field = "alpha.a"
        self.velocity_field = "U.b"
        self.alpha_threshold = 1e-5
        self.y_lower = 0.001
        self.x_lim = 3.0
        self.z_lim = None
        self.fig_size = (13, 7)
        self.curve_fig_size = (16, 7)
        self.cmap = "coolwarm"
        self.n_levels = 121
        self.rhoa = 3217.0
        self.rhob = 1000.0
        self.time_scale = 1.175
        self.head_x_scale = 0.3
        self.save_csv = True
        self.save_png = True

    @staticmethod
    def _time_to_dir_name(time_value: float) -> str:
        return f"{float(time_value):g}"

    @staticmethod
    def _reshape_field(field_flat, sort_idx, nx, ny, nz):
        n_cells = nx * ny * nz
        array = np.asarray(field_flat)
        if array.ndim == 1:
            if array.size == 1:
                array = np.full(n_cells, float(array[0]), dtype=float)
            if array.size != n_cells:
                raise ValueError(f"Scalar size {array.size} does not match {n_cells} cells")
            return array[sort_idx].reshape(nx, ny, nz)

        if array.ndim == 2:
            if array.shape == (3, 1):
                array = np.repeat(array, n_cells, axis=1)
            elif array.shape == (1, 3):
                array = np.repeat(array.T, n_cells, axis=1)
            if array.shape == (n_cells, 3):
                array = array.T
            if array.shape == (3, n_cells):
                return array[:, sort_idx].reshape(3, nx, ny, nz)

        raise ValueError(f"Unsupported field shape: {array.shape}")

    def _build_sorted_mesh(self):
        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        nx = len(np.unique(x_raw))
        ny = len(np.unique(y_raw))
        nz = len(np.unique(z_raw))
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))
        x_3d = self._reshape_field(x_raw, sort_idx, nx, ny, nz)
        y_3d = self._reshape_field(y_raw, sort_idx, nx, ny, nz)
        z_3d = self._reshape_field(z_raw, sort_idx, nx, ny, nz)
        x_axis = np.mean(x_3d, axis=2)[:, 0]
        y_axis = np.mean(y_3d, axis=2)[0, :]
        z_axis = np.mean(z_3d, axis=(0, 1))
        return sort_idx, nx, ny, nz, x_axis, y_axis, z_axis

    def _read_vector_3d(self, time_dir, field_name, sort_idx, nx, ny, nz):
        field = fluidfoam.readvector(self.sol, time_dir, field_name)
        vector = self._reshape_field(field, sort_idx, nx, ny, nz)
        if vector.shape != (3, nx, ny, nz):
            raise ValueError(f"{field_name} is not a three-component vector field: {vector.shape}")
        return vector.astype(float, copy=False)

    def _read_scalar_3d(self, time_dir, field_name, sort_idx, nx, ny, nz):
        field = fluidfoam.readscalar(self.sol, time_dir, field_name)
        scalar = self._reshape_field(field, sort_idx, nx, ny, nz)
        if scalar.shape != (nx, ny, nz):
            raise ValueError(f"{field_name} is not a scalar field: {scalar.shape}")
        return scalar.astype(float, copy=False)

    @staticmethod
    def _head_index(alpha_3d, threshold):
        indices = np.flatnonzero(np.any(alpha_3d > threshold, axis=(1, 2)))
        return int(indices.max()) if indices.size else None

    def _vertical_upper_bound(self, ubx_column, y_axis):
        valid = np.isfinite(ubx_column) & np.isfinite(y_axis)
        y_valid = y_axis[valid]
        u_valid = ubx_column[valid]
        if y_valid.size == 0:
            return np.nan
        crossings = np.flatnonzero((u_valid[:-1] > 0.0) & (u_valid[1:] <= 0.0) & (y_valid[:-1] >= self.y_lower))
        return float(y_valid[crossings[0] + 1]) if crossings.size else float(y_valid[-1])

    def vertical_average(self, field_xy, y_axis, ubx_xy):
        """Average a z-averaged x-y field vertically along y (axis 1)."""
        nx = field_xy.shape[0]
        result = np.zeros(nx, dtype=float)
        heights = np.zeros(nx, dtype=float)
        for x_index in range(nx):
            values = field_xy[x_index]
            velocity = ubx_xy[x_index]
            valid = np.isfinite(values) & np.isfinite(velocity) & np.isfinite(y_axis)
            y_valid = y_axis[valid]
            values_valid = values[valid]
            velocity_valid = velocity[valid]
            upper = None
            for y_index in range(len(velocity_valid) - 1):
                if y_valid[y_index] < self.y_lower:
                    continue
                if velocity_valid[y_index] > 0.0 and velocity_valid[y_index + 1] <= 0.0:
                    upper = float(y_valid[y_index + 1])
                    break
            upper = float(y_valid[-1]) if upper is None else upper
            selected = (y_valid >= self.y_lower) & (y_valid <= upper)
            y_selected = y_valid[selected]
            values_selected = values_valid[selected]
            if y_selected.size < 2:
                continue
            height = float(y_selected[-1] - y_selected[0])
            heights[x_index] = height
            result[x_index] = np.trapz(values_selected, x=y_selected) / height if height > 1e-12 else 0.0
        return result, heights

    def vertical_average_xz(self, field_x_y_z, y_axis, ubx_x_y_z):
        """Average a scalar field along y independently at every (x, z)."""
        nx, _, nz = field_x_y_z.shape
        result = np.full((nx, nz), np.nan)
        for x_index in range(nx):
            for z_index in range(nz):
                values = field_x_y_z[x_index, :, z_index]
                velocity = ubx_x_y_z[x_index, :, z_index]
                valid = np.isfinite(values) & np.isfinite(velocity) & np.isfinite(y_axis)
                y_valid = y_axis[valid]
                values_valid = values[valid]
                velocity_valid = velocity[valid]
                upper = self._vertical_upper_bound(velocity_valid, y_valid)
                selected = (y_valid >= self.y_lower) & (y_valid <= upper)
                y_selected = y_valid[selected]
                values_selected = values_valid[selected]
                if y_selected.size < 2:
                    continue
                height = float(y_selected[-1] - y_selected[0])
                if height > 1e-12:
                    result[x_index, z_index] = np.trapz(values_selected, x=y_selected) / height
        return result

    def _dimensionless(self, field):
        return field / self.rhob * self.time_scale**2

    def _plot_components(self, x_head, values, field_label, out_path):
        plt.figure(figsize=(9, 7))
        for component, curve in values.items():
            plt.plot(x_head, curve, label=component, linewidth=2)
        plt.xlabel(r"$(x_{head}-x)/L_0$")
        plt.ylabel("y-averaged stretching")
        plt.title(f"{field_label}: z-average (axis 2), then y-average (axis 1)")
        plt.xlim(0.0, self.x_lim)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def _plot_xz(self, x_head, z_axis, field_xz, component, field_label, out_path):
        finite = field_xz[np.isfinite(field_xz)]
        if finite.size == 0:
            return
        bound = float(np.percentile(np.abs(finite), 99.0))
        bound = max(bound, 1e-30)
        levels = np.linspace(-bound, bound, self.n_levels)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        plt.figure(figsize=self.fig_size)
        contour = plt.contourf(x_head, z_axis, field_xz.T, levels=levels,
                               cmap=self.cmap, norm=norm, extend="both")
        plt.colorbar(contour, label=f"{field_label}_{component}, y-avg")
        plt.xlabel(r"$(x_{head}-x)/L_0$")
        plt.ylabel("z (spanwise)")
        plt.title(f"{field_label}: {component}, y-average at each (x,z)")
        plt.xlim(0.0, self.x_lim)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def _plot_z_rms(self, x_head, statistics, field_label, out_path):
        plt.figure(figsize=(10, 6))
        for component, (z_mean, z_rms) in statistics.items():
            plt.plot(x_head, z_mean, label=f"{component} z-mean", linewidth=2)
            plt.plot(x_head, z_rms, "--", label=f"{component} z-RMS", linewidth=1.8)
        plt.xlabel(r"$(x_{head}-x)/L_0$")
        plt.ylabel("y-averaged stretching")
        plt.title(f"{field_label}: spanwise mean and RMS")
        plt.xlim(0.0, self.x_lim)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def process_time_step(self, time_value, mesh):
        sort_idx, nx, ny, nz, x_axis, y_axis, z_axis = mesh
        time_dir = self._time_to_dir_name(time_value)
        output_dir = os.path.join(self.output_root, self.output_prefix, f"t{time_dir}")
        os.makedirs(output_dir, exist_ok=True)

        alpha_3d = self._read_scalar_3d(time_dir, self.alpha_field, sort_idx, nx, ny, nz)
        ub_3d = self._read_vector_3d(time_dir, self.velocity_field, sort_idx, nx, ny, nz)
        # Match vorticitytransport_ofcal.py: locate the head after z averaging.
        alpha_2d = np.mean(alpha_3d, axis=2)
        head_index = self._head_index(alpha_2d[:, :, None], self.alpha_threshold)
        if head_index is None:
            raise ValueError(f"No {self.alpha_field} value exceeds {self.alpha_threshold} at t={time_dir}")
        head_x = x_axis[head_index]
        selected = np.arange(head_index + 1)
        x_head = (head_x - x_axis[selected]) / self.head_x_scale
        selected = selected[x_head >= 0.0]
        x_head = x_head[x_head >= 0.0]

        for label, field_name in self.stretching_fields.items():
            vector_3d = self._dimensionless(self._read_vector_3d(time_dir, field_name, sort_idx, nx, ny, nz))
            component_data = {}
            xz_data = {}
            statistics = {}
            frame = {"x": x_axis[selected], "x_head": x_head}
            ubx_xy = np.mean(ub_3d[0], axis=2)

            for component_index, component in enumerate(("Sx", "Sy", "Sz")):
                field_xy = np.mean(vector_3d[component_index], axis=2)
                curve, _ = self.vertical_average(field_xy, y_axis, ubx_xy)
                component_data[component] = curve[selected]
                frame[f"{component}_yavg"] = curve[selected]

                # Three-dimensional diagnostic: retain z after y averaging.
                field_xz = self.vertical_average_xz(
                    vector_3d[component_index], y_axis, ub_3d[0]
                )[selected, :]
                z_mean = np.nanmean(field_xz, axis=1)
                z_rms = np.sqrt(np.nanmean((field_xz - z_mean[:, None]) ** 2, axis=1))
                xz_data[component] = field_xz
                statistics[component] = (z_mean, z_rms)
                frame[f"{component}_zmean"] = z_mean
                frame[f"{component}_zrms"] = z_rms
                if self.save_png:
                    self._plot_xz(
                        x_head, z_axis, field_xz, component, label,
                        os.path.join(output_dir, f"{label}_{component}_yavg_xz.png"),
                    )

            if self.save_csv:
                pd.DataFrame(frame).to_csv(os.path.join(output_dir, f"{label}_zfirst_yavg.csv"), index=False)
                x_grid, z_grid = np.meshgrid(x_axis[selected], z_axis, indexing="ij")
                xhead_grid, _ = np.meshgrid(x_head, z_axis, indexing="ij")
                rows = []
                for component, values in xz_data.items():
                    rows.append(pd.DataFrame({
                        "x": x_grid.ravel(),
                        "x_head": xhead_grid.ravel(),
                        "z": z_grid.ravel(),
                        "component": component,
                        "value_yavg": values.ravel(),
                    }))
                pd.concat(rows, ignore_index=True).to_csv(
                    os.path.join(output_dir, f"{label}_yavg_xz.csv"), index=False
                )

            if self.save_png:
                self._plot_components(x_head, component_data, label, os.path.join(output_dir, f"{label}_zfirst_yavg.png"))
                self._plot_z_rms(
                    x_head, statistics, label,
                    os.path.join(output_dir, f"{label}_yavg_z_mean_rms.png"),
                )
            print(f"Saved {label} at t={time_dir}: {output_dir}")

    def run(self):
        mesh = self._build_sorted_mesh()
        for time_value in self.times:
            self.process_time_step(time_value, mesh)


if __name__ == "__main__":
    VorticityStretching3DAnalyzer().run()
