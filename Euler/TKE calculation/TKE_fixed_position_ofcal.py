import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


@dataclass
class TimeStepFields3D:
    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_a: np.ndarray
    Ub: np.ndarray
    kb: np.ndarray


class TKEFixedPositionAnalyzer:
    """Extract height and height_raw time series at fixed head-relative positions."""

    def __init__(self):
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090327_12"
        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d09_0327_12_fixed_xdime1e5"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        # self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0428_4_fixed_xdime1e5"
        self.times = np.arange(0.5, 40, 0.5)

        self.alpha_threshold = 1e-3
        self.alpha_threshold_target = 1e-5
        self.H = 0.3
        self.U = 0.26
        self.target_x_dimes = [1.2,1.5,2]

        self.tke_height_smooth_sigma = 2.0
        self.save_time_series_png = True
        self.time_series_plot_size = (9, 4.8)
        self.legend_fontsize = 12
        self.title_fontsize = 16
        self.label_fontsize = 14
        self.tick_fontsize = 12

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _nondim_time(time_v: float) -> float:
        return float(time_v) * 0.85

    def _time_label(self, time_v: float) -> str:
        return rf"$t*={self._nondim_time(time_v):.2f}$"

    def _target_tag(self) -> str:
        values = "_".join(f"{target:.2f}".replace(".", "p") for target in self.target_x_dimes)
        return f"xdime{values}"

    @staticmethod
    def _single_target_tag(target: float) -> str:
        return f"xdime{float(target):.2f}".replace(".", "p")

    @staticmethod
    def _build_grid_cache(X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray) -> Dict[str, np.ndarray]:
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
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
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
    def _vertical_average_to_zerocity_zero(
        field2d: np.ndarray,
        y_coords: np.ndarray,
        ubx2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            u_sel = u_valid[active_mask]

            if y_sel.size < 2:
                heights[i] = 0.0
                out[i] = 0.0
                continue

            if hasattr(np, "trapezoid"):
                numerator = float(np.trapezoid(f_sel, x=y_sel))
                int_u = float(np.trapezoid(u_sel, x=y_sel))
                int_u2 = float(np.trapezoid(u_sel**2, x=y_sel))
            else:
                numerator = float(np.trapz(f_sel, x=y_sel))
                int_u = float(np.trapz(u_sel, x=y_sel))
                int_u2 = float(np.trapz(u_sel**2, x=y_sel))

            velocity_height = int_u**2 / int_u2 if abs(int_u2) > 1e-20 else 0.0
            heights[i] = velocity_height
            if velocity_height > 1e-12:
                out[i] = numerator / velocity_height
            else:
                out[i] = 0.0

        return out, heights

    def _alpha_threshold_height(self, alpha_2d: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        heights = np.full(alpha_2d.shape[0], np.nan, dtype=float)

        for i in range(alpha_2d.shape[0]):
            alpha_profile = alpha_2d[i]
            valid = np.isfinite(alpha_profile) & np.isfinite(y_coords)
            y_valid = y_coords[valid]
            alpha_valid = alpha_profile[valid]
            above = alpha_valid >= self.alpha_threshold_target

            if y_valid.size == 0 or not np.any(above):
                continue

            top_idx = int(np.where(above)[0].max())
            if top_idx >= y_valid.size - 1:
                heights[i] = float(y_valid[top_idx])
                continue

            a0 = float(alpha_valid[top_idx])
            a1 = float(alpha_valid[top_idx + 1])
            y0 = float(y_valid[top_idx])
            y1 = float(y_valid[top_idx + 1])
            if abs(a1 - a0) > 1e-20:
                frac = (self.alpha_threshold_target - a0) / (a1 - a0)
                heights[i] = y0 + frac * (y1 - y0)
            else:
                heights[i] = y0

        return heights

    @staticmethod
    def _smooth_1d_nanaware(values: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.0:
            return values.copy()

        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            return values.copy()

        weights = gaussian_filter1d(valid.astype(float), sigma=sigma, mode="nearest")
        smoothed = gaussian_filter1d(np.where(valid, values, 0.0), sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = smoothed / weights
        smoothed[weights < 1e-8] = np.nan
        return smoothed

    @staticmethod
    def _trim_x_dime(x_seg: np.ndarray, x_head: float, H: float, x_dime_max: float):
        x_dime = (x_head - x_seg) / H
        mask = (x_dime >= 0.0) & (x_dime <= x_dime_max)
        return x_seg[mask], x_dime[mask], mask

    def _load_fields_3d(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepFields3D]:
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
            ub_raw = fluidfoam.readvector(self.sol, time_dir, "U.b")
            Ub = self._reshape_sorted(ub_raw, sort_idx, nx, ny, nz)
        except Exception as exc:
            print(f"Read failed for U.b at t={time_v}: {exc}")
            return None

        try:
            kb_raw = fluidfoam.readscalar(self.sol, time_dir, "k.b")
            kb = self._reshape_sorted(kb_raw, sort_idx, nx, ny, nz)
        except Exception as exc:
            print(f"Read failed for k.b at t={time_v}: {exc}")
            return None

        return TimeStepFields3D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            alpha_a=alpha_a,
            Ub=Ub,
            kb=kb,
        )

    @staticmethod
    def _interp_curve_at_target(
        x_plot: np.ndarray,
        values: np.ndarray,
        target_x_dime: float,
        valid_x: np.ndarray,
    ) -> float:
        valid = valid_x & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            return np.nan

        order = np.argsort(x_plot[valid])
        return float(np.interp(
            target_x_dime,
            x_plot[valid][order],
            values[valid][order],
        ))

    def _sample_at_fixed_positions(self, x_axis: np.ndarray, y_axis: np.ndarray, alpha_2d: np.ndarray, ubx_2d: np.ndarray, kb_2d: np.ndarray, head_idx: int, head_x: float) -> List[Dict[str, float]]:
        x_seg = x_axis[: head_idx + 1]
        x_dime = (head_x - x_seg) / self.H
        y_vals = y_axis

        alpha_seg = np.maximum(alpha_2d[: head_idx + 1, :], 0.0)
        kb_seg = kb_2d[: head_idx + 1, :]

        x_order = np.argsort(x_dime)
        x_plot = x_dime[x_order]
        alpha_plot = alpha_seg[x_order, :]
        kb_plot = kb_seg[x_order, :]

        if x_plot.size < 2 or np.any(np.diff(x_plot) <= 0):
            return [
                {"x_dime_target": float(target), "x_dime_sampled": np.nan}
                for target in self.target_x_dimes
            ]
        if y_vals.size < 2 or np.any(np.diff(y_vals) <= 0):
            return [
                {"x_dime_target": float(target), "x_dime_sampled": np.nan}
                for target in self.target_x_dimes
            ]

        _, kb_height = self._vertical_average_to_zerocity_zero(kb_plot, y_vals, ubx_2d[: head_idx + 1, :][x_order, :])
        height_raw_sel = kb_height / self.H
        height_sel = self._smooth_1d_nanaware(kb_height, self.tke_height_smooth_sigma) / self.H
        alpha_threshold_height_sel = self._alpha_threshold_height(alpha_plot, y_vals) / self.H

        valid_x = np.isfinite(x_plot)
        x_valid = x_plot[valid_x]
        if x_valid.size < 2:
            return [
                {"x_dime_target": float(target), "x_dime_sampled": np.nan}
                for target in self.target_x_dimes
            ]

        x_min = float(np.nanmin(x_valid))
        x_max = float(np.nanmax(x_valid))
        rows: List[Dict[str, float]] = []
        for target in self.target_x_dimes:
            target = float(target)
            out: Dict[str, float] = {
                "x_dime_target": target,
                "x_dime_sampled": np.nan,
                "x_dime_min": x_min,
                "x_dime_max": x_max,
            }

            if not (x_min <= target <= x_max):
                out["height_raw"] = np.nan
                out["height"] = np.nan
                out["alpha_threshold_height"] = np.nan
                rows.append(out)
                continue

            out["x_dime_sampled"] = target
            out["height_raw"] = self._interp_curve_at_target(x_plot, height_raw_sel, target, valid_x)
            out["height"] = self._interp_curve_at_target(x_plot, height_sel, target, valid_x)
            out["alpha_threshold_height"] = self._interp_curve_at_target(
                x_plot,
                alpha_threshold_height_sel,
                target,
                valid_x,
            )
            rows.append(out)

        return rows

    def _save_time_series_plot(self, df_series: pd.DataFrame) -> None:
        if not self.save_time_series_png or df_series.empty:
            return

        fig, ax = plt.subplots(figsize=self.time_series_plot_size)
        for target, df_target in df_series.groupby("x_dime_target", sort=True):
            t_star = df_target["t_star"].to_numpy(dtype=float)
            suffix = rf"$x_d/H={target:.2f}$"
            ax.plot(t_star, df_target["height_raw"], linestyle="--", linewidth=1.8, label=rf"$h_{{raw}}/H$, {suffix}")
            ax.plot(t_star, df_target["height"], linestyle="-", linewidth=1.8, label=rf"$h/H$, {suffix}")
            if "alpha_threshold_height" in df_target.columns:
                ax.plot(
                    t_star,
                    df_target["alpha_threshold_height"],
                    linestyle=":",
                    linewidth=1.8,
                    label=rf"$\alpha_a={self.alpha_threshold:g}$ height, {suffix}",
                )
        ax.set_xlabel(r"$t^*$", fontsize=self.label_fontsize)
        ax.set_ylabel(r"$h/H$", fontsize=self.label_fontsize)
        ax.set_title("Fixed-position height time series", fontsize=self.title_fontsize)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        ax.legend(fontsize=self.legend_fontsize, loc="best")
        fig.tight_layout()

        out_path = os.path.join(self.output_dir, f"fixed_position_series_{self._target_tag()}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Figure: {out_path}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> List[Dict[str, float]]:
        time_label = self._time_label(time_v)
        data_3d = self._load_fields_3d(grid, float(time_v))
        if data_3d is None:
            return []

        alpha_a_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            print(f"No alpha.a > threshold ({self.alpha_threshold}) at {time_label}. Skip output.")
            return []

        head_x = float(data_3d.x_axis[head_idx])
        print(f"Head position: x={head_x:.4f} (idx={head_idx}) at {time_label}")

        ubx_2d = np.mean(data_3d.Ub[0], axis=2)
        kb_2d = np.mean(data_3d.kb, axis=2)

        samples = self._sample_at_fixed_positions(
            data_3d.x_axis,
            data_3d.y_axis,
            alpha_a_2d,
            ubx_2d,
            kb_2d,
            head_idx,
            head_x,
        )
        for sample in samples:
            sample["time"] = float(time_v)
            sample["t_star"] = float(self._nondim_time(time_v))
            sample["head_x"] = head_x
            sample["x_target"] = head_x - float(sample["x_dime_target"]) * self.H
            sample["head_idx"] = int(head_idx)
        return samples

    def run_analysis(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        rows = []
        for t in self.times:
            rows.extend(self.process_time_step(grid, float(t)))

        if not rows:
            print("No valid rows were extracted. Nothing to save.")
            return

        df_series = pd.DataFrame(rows).sort_values(["time", "x_dime_target"]).reset_index(drop=True)
        for target, df_target in df_series.groupby("x_dime_target", sort=True):
            csv_path = os.path.join(self.output_dir, f"fixed_position_series_{self._single_target_tag(target)}.csv")
            df_target = df_target.sort_values("time").reset_index(drop=True)
            df_target.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")

        self._save_time_series_plot(df_series)


if __name__ == "__main__":
    analyzer = TKEFixedPositionAnalyzer()
    analyzer.run_analysis()
