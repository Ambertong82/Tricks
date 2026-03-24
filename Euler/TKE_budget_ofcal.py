import os
from dataclasses import dataclass
from typing import Dict, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class TimeStepTerms3D:
    """Container for one time-step fields used in postprocessing only.

    All arrays are in the reconstructed structured-grid layout (nx, ny, nz),
    not the original flattened OpenFOAM ordering.
    """

    time: float
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    alpha_a: np.ndarray
    terms: Dict[str, np.ndarray]


class TKEBudgetAnalyzer:
    def __init__(self):
        # OpenFOAM case directory and output directory for generated CSV/figures.
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230311_2_1"
        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0311_2_1"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/case090311_10"
        # self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d09_0311_10"
        # Physical times to be processed.
        self.times = [15, 25, 34]

        # Threshold to detect current head position from alpha.a.
        self.alpha_threshold = 1e-5
        # Default style settings shared by all figures.
        self.fig_size = (20, 3)
        self.curve_lw = 2.0
        # Variables compared across all selected times in one summary figure set.
        # You can use either raw names (e.g. "G") or curve-column names (e.g. "G_avg").
        self.comparison_columns = ["G", "convection", "diff", "drag1", "dissipation", "dkdtof"]
        # Set None to use all variables from comparison_columns.
        # Set an integer (e.g. 2) to only output the first N variables.
        self.num_comparison_variables = None

        # Mapping: output column name -> OpenFOAM field name.
        # Keep/edit this list based on what has already been computed in OF.
        self.of_term_fields = {
            "convection": "Kconvection",
            "G": "Kprod",
            "density_gradient": "Kgrad",
            "dissipation": "Kdissip",
            "drag1": "drag1",
            "drag2": "drag2",
            "drag3": "drag3",
            "dkdtof": "dkdt",
            "diff": "Kdiff1",
            "ksource": "Ksource",
            "kresidual": "Kresidual",
        }

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        """Format float time to match OpenFOAM folder names like 16, 15.5, 16.25.

        OpenFOAM time folders are usually compact (no trailing zeros), so
        `:g` formatting keeps compatibility with folder naming.
        """
        return f"{float(time_v):g}"

    @staticmethod
    def _build_grid_cache(X_raw: np.ndarray, Y_raw: np.ndarray, Z_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Precompute mesh metadata used to reshape flattened OpenFOAM fields.

        OpenFOAM read functions return flattened arrays. We lexicographically sort
        by (X, Y, Z), then reshape to consistent (nx, ny, nz) structured arrays.
        """
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        # Sort by x first, then y, then z for deterministic reconstruction.
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
        """Reshape flattened OpenFOAM data after lexsort; reconstruction order is C.

        Supports scalar fields with shape (N,) and vector-like fields with shape
        (ncomp, N). In this script we mainly use scalar fields.
        """
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    @staticmethod
    def _vertical_average(field2d: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Depth-average along y for each x, producing 1D curve vs x.

        field2d shape is (nx, ny). Integration is along y (axis=1).
        """
        if hasattr(np, "trapezoid"):
            numerator = np.trapezoid(field2d, x=y_coords, axis=1)
        else:
            numerator = np.trapz(field2d, x=y_coords, axis=1)

        depth = y_coords[-1] - y_coords[0]
        return np.divide(numerator, depth, out=np.zeros_like(numerator), where=np.abs(depth) > 1e-12)

    def _locate_head_index(self, alpha_a_2d: np.ndarray) -> Optional[int]:
        """Find last x index where spanwise-averaged alpha_a exceeds threshold at any y.

        This x index is used as the front/head location; all curves are truncated
        to [0, head_idx] so only the current body/head region is plotted.
        """
        mask_x = np.any(alpha_a_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if len(valid_x) == 0:
            return None
        return int(valid_x.max())

    def _load_terms_3d(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[TimeStepTerms3D]:
        """Load OF-precomputed budget terms for one time.

        Important: this script does not derive terms via gradients; it only reads
        fields that were already computed and written by OpenFOAM.
        """
        print(f"\n>>> Processing time: {time_v}")
        time_dir = self._time_to_dir_name(time_v)

        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]

        try:
            alpha_a_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
        except Exception as exc:
            print(f"Read failed for alpha.a at t={time_v}: {exc}")
            return None

        alpha_a = self._reshape_sorted(alpha_a_raw, sort_idx, nx, ny, nz)

        # Read each configured budget term if the field exists at this time.
        loaded_terms: Dict[str, np.ndarray] = {}
        for out_name, of_name in self.of_term_fields.items():
            try:
                term_raw = fluidfoam.readscalar(self.sol, time_dir, of_name)
                loaded_terms[out_name] = self._reshape_sorted(term_raw, sort_idx, nx, ny, nz)
            except Exception as exc:
                print(f"Skip missing term '{of_name}' at t={time_v}: {exc}")

        if not loaded_terms:
            print(f"No budget terms were loaded at t={time_v}. Skip output.")
            return None

        return TimeStepTerms3D(
            time=float(time_v),
            x_axis=grid["x_axis_3d"],
            y_axis=grid["y_axis_3d"],
            z_axis=grid["z_axis_3d"],
            alpha_a=alpha_a,
            terms=loaded_terms,
        )

    @staticmethod
    def _spanwise_average_terms(terms_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Average each term along z to obtain unified 2D terms (x, y)."""
        return {name: np.mean(field, axis=2) for name, field in terms_3d.items()}

    def _average_to_curves(
        self,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        terms_2d: Dict[str, np.ndarray],
        head_idx: int,
    ) -> pd.DataFrame:
        # Keep only data up to the detected head location.
        x_seg = x_axis[: head_idx + 1]
        x_head = x_axis[head_idx]

        curves = {
            "x": x_seg,
            "x_dime": (x_head - x_seg) / 0.3,
        }

        for name, field in terms_2d.items():
            # Convert each 2D field to a 1D depth-averaged curve vs x.
            curve = self._vertical_average(field, y_axis)
            curves[f"{name}_avg"] = curve[: head_idx + 1]

        return pd.DataFrame(curves)

    @staticmethod
    def _format_plot_label(column_name: str) -> str:
        """Map output column names to MathText labels for legends.

        Unknown names fall back to raw column strings so new terms still plot.
        """
        label_map = {
            "convection_avg": r"$\langle C \rangle_d$",
            "G_avg": r"$\langle G \rangle_d$",
            "density_gradient_avg": r"$\langle \nabla \rho \rangle_d$",
            "dissipation_avg": r"$\langle \epsilon \rangle_d$",
            "drag1_avg": r"$\langle Drag_{d1} \rangle_d$",
            "drag2_avg": r"$\langle Drag_{d2} \rangle_d$",
            "drag3_avg": r"$\langle Drag_{d3} \rangle_d$",
            "ksource_avg": r"$\langle RHS \rangle_d$",
            "dkdtof_avg": r"$\langle (\rho\beta k)_t \rangle_d$",
            "diff_avg": r"$\langle D \rangle_d$",
            
        }
        return label_map.get(column_name, column_name)

    @staticmethod
    def _legend_if_any(ax, **kwargs):
        """Only draw a legend when at least one labeled line exists."""
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)

    @staticmethod
    def _as_curve_column(name: str) -> str:
        """Convert a variable key to its curve column name, e.g. G -> G_avg.

        This allows users to configure comparison_columns with or without _avg.
        """
        return name if name.endswith("_avg") else f"{name}_avg"

    def _save_outputs(self, time_v: float, df_curve: pd.DataFrame):
        """Save per-time CSV and three default summary figures."""
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_v:.2f}.csv")
        df_curve.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # Figure 1: almost all terms except selected bookkeeping terms.
        fig, ax = plt.subplots(figsize=self.fig_size)
        for col in df_curve.columns:
            if col in ("x", "x_dime", "dkdtof_avg", "ksource_avg", "kresidual_avg"):
                continue
            linestyle = "--" if col in ("drag1_avg", "drag2_avg", "drag3_avg") else "-"
            ax.plot(
                df_curve["x_dime"],
                df_curve[col],
                linewidth=self.curve_lw,
                linestyle=linestyle,
                label=self._format_plot_label(col),
            )

        ax.set_title(f"TKE Budget Terms (Vertical Average) at t={time_v:.2f}s", fontsize=16)
        ax.set_xlabel(r"$(x_f-x)/H$", fontsize=14)
        ax.set_xlim(df_curve["x_dime"].max(), 0.0)
        ax.set_ylabel("Vertical average", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)
        self._legend_if_any(ax, fontsize=10, ncol=2, loc="upper left")
        fig.tight_layout()

        fig_path = os.path.join(self.output_dir, f"TKE_Budget_SpanAvg_VAvg_t{time_v:.2f}.png")
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved Figure: {fig_path}")

        # Figure 2: residual-only diagnostic.
        fig2, ax2 = plt.subplots(figsize=self.fig_size)
        for col in ("kresidual_avg",):
            if col in df_curve.columns:
                ax2.plot(df_curve["x_dime"], df_curve[col], linewidth=self.curve_lw, label=self._format_plot_label(col))

        ax2.set_title(f"TKE Budget Terms summary at t={time_v:.2f}s", fontsize=16)
        ax2.set_xlabel(r"$(x_f-x)/H$", fontsize=14)
        ax2.set_xlim(df_curve["x_dime"].max(), 0.0)
        ax2.set_ylabel("Vertical average", fontsize=14)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.grid(True, linestyle="--", alpha=0.35)
        self._legend_if_any(ax2, fontsize=10, ncol=2, loc="upper left")
        fig2.tight_layout()

        fig2_path = os.path.join(self.output_dir, f"TKE_Budget_Summary_t{time_v:.2f}.png")
        fig2.savefig(fig2_path, dpi=300)
        plt.close(fig2)
        print(f"Saved Figure: {fig2_path}")

        # Figure 3: selected RHS-related terms.
        fig3, ax3 = plt.subplots(figsize=self.fig_size)
        for col in ("dkdtof_avg", "convection_avg", "ksource_avg"):
            if col in df_curve.columns:
                ax3.plot(df_curve["x_dime"], df_curve[col], linewidth=self.curve_lw, label=self._format_plot_label(col))

        ax3.set_title(f"TKE Budget Terms summary2 at t={time_v:.2f}s", fontsize=16)
        ax3.set_xlabel(r"$(x_f-x)/H$", fontsize=14)
        ax3.set_xlim(df_curve["x_dime"].max(), 0.0)
        ax3.set_ylabel("Vertical average", fontsize=14)
        ax3.tick_params(axis="both", labelsize=12)
        ax3.grid(True, linestyle="--", alpha=0.35)
        self._legend_if_any(ax3, fontsize=10, ncol=2, loc="upper left")
        fig3.tight_layout()

        fig3_path = os.path.join(self.output_dir, f"TKE_Budget_Summary2_t{time_v:.2f}.png")
        fig3.savefig(fig3_path, dpi=300)
        plt.close(fig3)
        print(f"Saved Figure: {fig3_path}")

    def _save_comparison(self, comparison_frames):
        """Save separate figures for selected variables across all time steps.

        Each output figure contains one variable and multiple time curves.
        """
        if not comparison_frames:
            print("No valid data found. Skip multi-variable comparison figure.")
            return

        # Normalize user requests to DataFrame column names.
        requested_columns = [self._as_curve_column(col) for col in self.comparison_columns]
        if self.num_comparison_variables is not None:
            # Limit to first N requests if user configured a cap.
            requested_columns = requested_columns[: max(0, int(self.num_comparison_variables))]

        available_columns = [
            col for col in requested_columns if any(col in df.columns for _, df in comparison_frames)
        ]
        if not available_columns:
            print("None of requested comparison columns exist in processed outputs. Skip comparison figure.")
            return

        for col in available_columns:
            fig, ax = plt.subplots(figsize=self.fig_size)
            xmax = 0.0
            for time_v, df_curve in comparison_frames:
                if col not in df_curve.columns:
                    continue
                xvals = df_curve["x_dime"].to_numpy()
                yvals = df_curve[col].to_numpy()
                ax.plot(xvals, yvals, linewidth=self.curve_lw, label=f"t={time_v:g}s")
                xmax = max(xmax, float(np.max(xvals)))

            ax.set_title(f"{self._format_plot_label(col)} Comparison Across Time Steps", fontsize=14)
            ax.set_xlabel(r"$(x_f-x)/H$", fontsize=14)
            ax.set_xlim(xmax, 0.0)
            ax.set_ylabel("Vertical average", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True, linestyle="--", alpha=0.35)
            self._legend_if_any(ax, fontsize=10, ncol=1, loc="upper left")
            fig.tight_layout()

            fig_path = os.path.join(self.output_dir, f"TKE_Comparison_{col}.png")
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            print(f"Saved Figure: {fig_path}")

    def process_time_step(self, grid: Dict[str, np.ndarray], time_v: float) -> Optional[pd.DataFrame]:
        """Process one time step and return the depth-averaged curve table."""
        data_3d = self._load_terms_3d(grid, float(time_v))
        if data_3d is None:
            return None

        # Use spanwise-averaged alpha_a only for front/head detection.
        alpha_a_2d = np.mean(data_3d.alpha_a, axis=2)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            print(f"No alpha.a > threshold ({self.alpha_threshold}) at t={time_v}. Skip output.")
            return None

        head_x = data_3d.x_axis[head_idx]
        print(f"Head position: x={head_x:.4f} (idx={head_idx})")

        terms_2d = self._spanwise_average_terms(data_3d.terms)
        df_curve = self._average_to_curves(data_3d.x_axis, data_3d.y_axis, terms_2d, head_idx)
        self._save_outputs(float(time_v), df_curve)
        return df_curve

    def run_analysis(self):
        """Main entry: read mesh once, process all times, then save cross-time comparisons."""
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        comparison_frames = []
        for t in self.times:
            df_curve = self.process_time_step(grid, t)
            if df_curve is None:
                continue
            comparison_frames.append((float(t), df_curve))

        self._save_comparison(comparison_frames)


if __name__ == "__main__":
    analyzer = TKEBudgetAnalyzer()
    analyzer.run_analysis()
