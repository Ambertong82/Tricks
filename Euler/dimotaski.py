import os
from typing import Dict, List, Optional

import fluidfoam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, TwoSlopeNorm


class TurbidityCurrentAnalyzer:
    def __init__(self):
        # OpenFOAM case
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230311_1"
        self.output_dir = "/home/amber/postpro/u_vorticity/tc3d_23ofcal"
        self.paraview_dir = os.path.join(self.output_dir, "paraview")
        self.times = [12,12.5,13]

        # 仅读取已经算好的项
        self.vort_fields: Dict[str, str] = {
            "U.b": "U.b",
            "alpha.a": "alpha.a",
            
        }

        self.fig_size = (20, 6)
        self.cmap = "coolwarm"
        self.n_levels = 121
        self.x_lim = None
        self.y_lim = (0.0, 0.30)
        self.alpha_threshold = 1e-5
        self.head_x_scale = 0.3
        self.clip_negative_x = True
        self.robust_percentile = (1.0, 99.0)
        self.advection_percentile = (3.0, 92.0)
        self.advection_gamma = 0.45
        self.diffusion_percentile = (5.0, 90.0)
        self.diffusion_gamma = 0.35
        self.export_paraview = True
        self.rhoa = 3217
        self.rhob = 1000
        self.H0 = 0.3
        self.g = 9.81
        self.n_vortex_cores = 6
        self.q_core_percentile = 95.0
        self.min_core_separation = 4
        self.q_core_sign = "auto"
        self.q_core_use_local_extrema = True
        self.q_core_alpha_threshold = 1e-5
        # 已知点（物理坐标）: 在这里填入你要提取 u1/u2 的点
        # 示例: {"name": "P1", "x_physical": 1.2, "y": 0.06}
        self.known_points: List[dict] = []
        # 若最近网格点与目标点距离超过该阈值(米)，仍写入但标记 matched=False
        self.known_point_max_distance = 0.02

    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _reshape_field(field_flat: np.ndarray, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        n_cells = nx * ny * nz
        arr = np.asarray(field_flat)

        if sort_idx.size != n_cells:
            raise ValueError(f"sort_idx size mismatch: got {sort_idx.size}, expected {n_cells}")

        if arr.ndim == 1:
            if arr.size != n_cells:
                raise ValueError(f"field size mismatch: got {arr.size}, expected {n_cells}")
            return arr[sort_idx].reshape(nx, ny, nz)

        if arr.ndim == 2:
            # fluidfoam vector layout can be (3, n_cells) or (n_cells, 3)
            if arr.shape == (n_cells, 3):
                arr = arr.T
            if arr.shape == (3, n_cells):
                return arr[:, sort_idx].reshape(3, nx, ny, nz)

        raise ValueError(f"Unsupported field shape {arr.shape}; expected (n_cells,), (3, n_cells) or (n_cells, 3)")

    @staticmethod
    def compute_spanwise_average(field_3d: np.ndarray) -> np.ndarray:
        # Spanwise direction is the last grid axis (z) after reshape.
        return np.mean(field_3d, axis=-1)

    @staticmethod
    def vector_to_x_component_2d(vector_2d: np.ndarray) -> np.ndarray:
        # Extract x component from a (3, nx, ny) vector field.
        if vector_2d.ndim != 3 or vector_2d.shape[0] != 3:
            raise ValueError(f"Expected vector_2d shape (3, nx, ny), got {vector_2d.shape}")
        return vector_2d[0, :, :]

    @staticmethod
    def compute_2D_Q_criterion(u_avg: np.ndarray, v_avg: np.ndarray, x_1d: np.ndarray, y_1d: np.ndarray) -> np.ndarray:
        """
        基于展向平均后的 2D 速度场计算二维 Q 准则。
        必须传入真实的 1D 物理坐标数组，以确保导数量级正确！
        u_avg, v_avg 形状为 (nx, ny)
        """
        # np.gradient 中，axis=0 对应行(nx/x向), axis=1 对应列(ny/y向)
        # 传入 x_1d 和 y_1d 强制让 Numpy 使用真实的物理网格间距 (dx, dy)
        du_dx, du_dy = np.gradient(u_avg, x_1d, y_1d, axis=(0, 1))
        dv_dx, dv_dy = np.gradient(v_avg, x_1d, y_1d, axis=(0, 1))
        
        # 二维不可压缩流体的 Q 准则简化绝对公式：
        # Q = - (du_dx * dv_dy) + (du_dy * dv_dx)
        Q_2d = (du_dx * dv_dy) - (du_dy * dv_dx)
        
        return Q_2d


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

    def _to_head_frame_x(self, x_2d: np.ndarray, head_x: float) -> np.ndarray:
        if self.head_x_scale == 0.0:
            raise ValueError("head_x_scale must be non-zero")
        return (head_x - x_2d) / self.head_x_scale

    def _build_sorted_mesh(self):
        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        nx, ny, nz = len(np.unique(x_raw)), len(np.unique(y_raw)), len(np.unique(z_raw))
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))

        x_3d = self._reshape_field(x_raw, sort_idx, nx, ny, nz)
        y_3d = self._reshape_field(y_raw, sort_idx, nx, ny, nz)

        x_2d = self.compute_spanwise_average(x_3d)
        y_2d = self.compute_spanwise_average(y_3d)
        return sort_idx, nx, ny, nz, x_2d, y_2d

    def _read_vector_3d(self, time_dir: str, field_name: str, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        vector_flat = fluidfoam.readvector(self.sol, time_dir, field_name)
        vector_3d = self._reshape_field(vector_flat, sort_idx, nx, ny, nz)
        if vector_3d.ndim != 4 or vector_3d.shape[0] != 3:
            raise ValueError(f"Field {field_name} is not a 3-component vector after reshape: {vector_3d.shape}")
        return vector_3d

    def _read_scalar_3d(self, time_dir: str, field_name: str, sort_idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
        scalar_flat = fluidfoam.readscalar(self.sol, time_dir, field_name)
        scalar_3d = self._reshape_field(scalar_flat, sort_idx, nx, ny, nz)
        if scalar_3d.ndim != 3:
            raise ValueError(f"Field {field_name} is not scalar after reshape: {scalar_3d.shape}")
        return scalar_3d

    def _plot_contour(
        self,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        q_2d: np.ndarray,
        title: str,
        out_path: str,
        percentile: tuple = None,
        gamma: float = None,
    ) -> None:
        q_plot = np.array(q_2d, copy=True)
        if self.clip_negative_x:
            q_plot = np.where(x_2d < 0.0, np.nan, q_plot)

        q_valid = q_plot[np.isfinite(q_plot)]
        if q_valid.size == 0:
            levels = self.n_levels
            norm = None
        else:
            if percentile is None:
                p_low, p_high = self.robust_percentile
            else:
                p_low, p_high = percentile
            vmin = float(np.percentile(q_valid, p_low))
            vmax = float(np.percentile(q_valid, p_high))

            # Fallback when field is nearly constant.
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(q_valid))
                vmax = float(np.nanmax(q_valid))

            if vmax <= vmin:
                levels = self.n_levels
                norm = None
            elif gamma is not None and gamma > 0.0 and vmin >= 0.0:
                levels = np.linspace(vmin, vmax, self.n_levels)
                norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
            elif vmin < 0.0 < vmax:
                levels = np.linspace(vmin, vmax, self.n_levels)
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                levels = np.linspace(vmin, vmax, self.n_levels)
                norm = None

        plt.figure(figsize=self.fig_size)
        cf = plt.contourf(x_2d, y_2d, q_plot, levels=levels, cmap=self.cmap, norm=norm, extend="both")
        plt.colorbar(cf)

        if self.x_lim is None:
            if self.clip_negative_x:
                x_nonneg = x_2d[np.isfinite(x_2d) & (x_2d >= 0.0)]
                if x_nonneg.size > 0:
                    plt.xlim(float(np.nanmax(x_nonneg)), 0.0)
                else:
                    plt.xlim(float(np.nanmax(x_2d)), float(np.nanmin(x_2d)))
            else:
                plt.xlim(float(np.nanmax(x_2d)), float(np.nanmin(x_2d)))
        else:
            plt.xlim(*self.x_lim)
        plt.ylim(*self.y_lim)
        plt.xlabel("(head_x - x) / 0.3")
        plt.ylabel("y (m)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def _save_paraview_vtk(self, x_2d: np.ndarray, y_2d: np.ndarray, q_2d: np.ndarray, scalar_name: str, out_path: str) -> None:
        nx, ny = q_2d.shape
        z_2d = np.zeros_like(q_2d)

        with open(out_path, "w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Spanwise averaged field\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"POINTS {nx * ny} float\n")

            for j in range(ny):
                for i in range(nx):
                    f.write(f"{float(x_2d[i, j]):.9e} {float(y_2d[i, j]):.9e} {float(z_2d[i, j]):.9e}\n")

            f.write(f"POINT_DATA {nx * ny}\n")
            f.write(f"SCALARS {scalar_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    val = float(q_2d[i, j])
                    if not np.isfinite(val):
                        val = -9999.0
                    f.write(f"{val:.9e}\n")

    def _find_vortex_cores(
        self,
        q_2d: np.ndarray,
        x_headframe_2d: np.ndarray,
        x_physical_2d: np.ndarray,
        y_2d: np.ndarray,
        alpha_2d: np.ndarray,
        top_k: int,
        percentile: float,
        min_separation: int,
        sign_mode: str = "auto",
        use_local_extrema: bool = True,
        alpha_threshold: float = 0.0,
    ) -> List[dict]:
        q_work = np.array(q_2d, copy=True)
        q_work[~np.isfinite(q_work)] = np.nan
        if np.all(~np.isfinite(q_work)):
            return []

        alpha_mask = np.isfinite(alpha_2d) & (alpha_2d > alpha_threshold)
        valid_mask = np.isfinite(q_work) & alpha_mask
        if not np.any(valid_mask):
            return []

        q_valid = q_work[valid_mask]
        q_pos = q_valid[q_valid > 0.0]
        q_neg = q_valid[q_valid < 0.0]

        if sign_mode == "positive":
            use_positive = True
        elif sign_mode == "negative":
            use_positive = False
        else:
            strongest_pos = float(np.nanmax(q_pos)) if q_pos.size > 0 else -np.inf
            strongest_neg = float(np.nanmin(q_neg)) if q_neg.size > 0 else np.inf
            use_positive = abs(strongest_pos) >= abs(strongest_neg)

        if use_positive:
            candidate_base = q_work
            q_pool = q_valid[q_valid > 0.0]
            if q_pool.size == 0:
                return []
            q_threshold = float(np.percentile(q_pool, percentile))
            candidate_mask = valid_mask & (q_work >= q_threshold)
        else:
            candidate_base = -q_work
            q_pool = (-q_valid)[q_valid < 0.0]
            if q_pool.size == 0:
                return []
            q_threshold = float(np.percentile(q_pool, percentile))
            candidate_mask = valid_mask & (-q_work >= q_threshold)

        if use_local_extrema:
            local_mask = np.zeros_like(candidate_mask, dtype=bool)
            rows, cols = q_work.shape
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if not candidate_mask[i, j]:
                        continue
                    window = candidate_base[i - 1:i + 2, j - 1:j + 2]
                    center = candidate_base[i, j]
                    if not np.isfinite(center):
                        continue
                    if np.nanmax(window) == center:
                        local_mask[i, j] = True
            candidate_mask = local_mask

        ii, jj = np.where(candidate_mask)
        if ii.size == 0:
            return []

        scores = q_work[ii, jj]
        if use_positive:
            order = np.argsort(scores)[::-1]
        else:
            order = np.argsort(scores)

        selected = []
        for idx in order:
            i = int(ii[idx])
            j = int(jj[idx])
            keep = True
            for core in selected:
                di = i - core["i"]
                dj = j - core["j"]
                if di * di + dj * dj < min_separation * min_separation:
                    keep = False
                    break
            if not keep:
                continue

            selected.append(
                {
                    "i": i,
                    "j": j,
                    "x_headframe": float(x_headframe_2d[i, j]),
                    "x_physical": float(x_physical_2d[i, j]),
                    "y": float(y_2d[i, j]),
                    "q": float(q_work[i, j]),
                }
            )
            if len(selected) >= top_k:
                break

        return selected

    def _write_known_points_u1u2_csv(
        self,
        time_dir: str,
        x_2d: np.ndarray,
        y_2d: np.ndarray,
        x_headframe_2d: np.ndarray,
        u1_2d: np.ndarray,
        u2_2d: np.ndarray,
        uc_2d: np.ndarray,
    ) -> Optional[str]:
        if not self.known_points:
            return None

        out_csv = os.path.join(self.output_dir, f"known_points_u1u2_t{time_dir}.csv")
        with open(out_csv, "w", encoding="ascii") as f:
            f.write(
                "point_name,target_x,target_y,i,j,matched,distance,x_physical,x_headframe,y,u1,u2,uc\n"
            )

            for idx, pt in enumerate(self.known_points, start=1):
                name = str(pt.get("name", f"P{idx}"))
                if "x_physical" not in pt or "y" not in pt:
                    continue

                x_tar = float(pt["x_physical"])
                y_tar = float(pt["y"])

                d2 = (x_2d - x_tar) ** 2 + (y_2d - y_tar) ** 2
                if np.all(~np.isfinite(d2)):
                    continue

                flat_idx = int(np.nanargmin(d2))
                i, j = np.unravel_index(flat_idx, d2.shape)
                dist = float(np.sqrt(d2[i, j]))
                matched = dist <= self.known_point_max_distance

                f.write(
                    f"{name},{x_tar:.9e},{y_tar:.9e},{i},{j},{int(matched)},{dist:.9e},"
                    f"{float(x_2d[i, j]):.9e},{float(x_headframe_2d[i, j]):.9e},{float(y_2d[i, j]):.9e},"
                    f"{float(u1_2d[i, j]):.9e},{float(u2_2d[i, j]):.9e},{float(uc_2d[i, j]):.9e}\n"
                )

        return out_csv

    def process_time_step(self, time_v: float, sort_idx: np.ndarray, nx: int, ny: int, nz: int, x_2d: np.ndarray, y_2d: np.ndarray) -> None:
        time_dir = self._time_to_dir_name(time_v)
        print(f"Processing t={time_dir}...")

        alpha_a_3d = self._read_scalar_3d(time_dir, "alpha.a", sort_idx, nx, ny, nz)
        alpha_a_2d = self.compute_spanwise_average(alpha_a_3d)
        alpha_a_2d = np.maximum(alpha_a_2d, 0.0)
        head_idx = self._locate_head_index(alpha_a_2d)
        if head_idx is None:
            head_x = float(np.nanmax(x_2d))
            print(f"  Warning: no alpha.a > {self.alpha_threshold:g}, fallback head_x={head_x:.4g}")
        else:
            head_x = float(x_2d[head_idx, 0])
            print(f"  head_x={head_x:.4g} (idx={head_idx}, threshold={self.alpha_threshold:g})")
        x_plot_2d = self._to_head_frame_x(x_2d, head_x)

        # 1. 读取 3D 速度并立刻做展向平均 (过滤掉 3D 湍流高频脉动)
        ub_3d = self._read_vector_3d(time_dir, "U.b", sort_idx, nx, ny, nz)
        U_avg_3d = self.compute_spanwise_average(ub_3d) # 形状变为 (3, nx, ny)
        
        # 2. 提取二维的 X 和 Y 速度分量
        ub_2d = U_avg_3d[0, :, :] # u 速度 (用于你下面的积分算 U1, U2)
        vb_2d = U_avg_3d[1, :, :] # v 速度

        y_line = y_2d[0, :]
        # 方案A: 下限从 y~0 到 cross_idx
        u1_by_x_y0 = np.full(nx, np.nan, dtype=float)
        u2_by_x_y0 = np.full(nx, np.nan, dtype=float)
        uc_by_x_y0 = np.full(nx, np.nan, dtype=float)
        alpha1_by_x_y0 = np.full(nx, np.nan, dtype=float)
        alpha2_by_x_y0 = np.full(nx, np.nan, dtype=float)

        # 方案B: 下限从剖面峰值 start_idx 到 cross_idx
        u1_by_x = np.full(nx, np.nan, dtype=float)
        u2_by_x = np.full(nx, np.nan, dtype=float)
        uc_by_x = np.full(nx, np.nan, dtype=float)
        alpha1_by_x = np.full(nx, np.nan, dtype=float)
        alpha2_by_x = np.full(nx, np.nan, dtype=float)

        for i in range(nx):
            ub_line = ub_2d[i, :]
            alpha_line = alpha_a_2d[i, :]

            sign_changes = np.where(np.diff(np.sign(ub_line)))[0]
            cross_idx = None
            for idx in sign_changes:
                if y_line[idx] > 0.001 and ub_line[idx] > 0 and ub_line[idx + 1] < 0:
                    cross_idx = idx + 1
                    break

            if cross_idx is None:
                continue

            finite_mask = np.isfinite(ub_line)
            if not np.any(finite_mask):
                continue

            start_idx = int(np.nanargmax(np.where(finite_mask, ub_line, np.nan)))
            if start_idx >= cross_idx:
                continue

            # ---------- 方案A: y~0 -> cross_idx ----------
            y_lower_y0 = y_line[:cross_idx + 1]
            ub_lower_y0 = ub_line[:cross_idx + 1]
            y_upper = y_line[cross_idx:]
            ub_upper = ub_line[cross_idx:]

            int_u_lower_y0 = np.trapz(ub_lower_y0, y_lower_y0)
            int_u2_lower_y0 = np.trapz(ub_lower_y0**2, y_lower_y0)
            int_u_upper_y0 = np.trapz(ub_upper, y_upper)
            int_u2_upper_y0 = np.trapz(ub_upper**2, y_upper)

            int_alpha_lower_y0 = np.trapz(alpha_line[:cross_idx + 1], y_lower_y0)
            int_alphaU_lower_y0 = np.trapz(alpha_line[:cross_idx + 1] * ub_lower_y0, y_lower_y0)
            int_alpha_upper_y0 = np.trapz(alpha_line[cross_idx:], y_upper)
            int_alphaU_upper_y0 = np.trapz(alpha_line[cross_idx:] * ub_upper, y_upper)

            u1_y0 = int_u2_lower_y0 / int_u_lower_y0 if int_u_lower_y0 != 0 else np.nan
            u2_y0 = int_u2_upper_y0 / int_u_upper_y0 if int_u_upper_y0 != 0 else np.nan
            alpha1_y0 = int_alphaU_lower_y0 / int_alpha_lower_y0 if int_alpha_lower_y0 != 0 else np.nan
            alpha2_y0 = int_alphaU_upper_y0 / int_alpha_upper_y0 if int_alpha_upper_y0 != 0 else np.nan
            alpha2_y0 = min(max(alpha2_y0, 0.0), 1.0)

            u1_by_x_y0[i] = u1_y0
            u2_by_x_y0[i] = u2_y0
            alpha1_by_x_y0[i] = alpha1_y0
            alpha2_by_x_y0[i] = alpha2_y0

            r_y0 = (alpha2_y0 * self.rhoa + (1.0 - alpha2_y0) * self.rhob) / (
                alpha1_y0 * self.rhoa + (1.0 - alpha1_y0) * self.rhob
            )
            sqrt_r_y0 = np.sqrt(r_y0) if r_y0 > 0 else np.nan
            if np.isfinite(sqrt_r_y0) and np.isfinite(u1_y0) and np.isfinite(u2_y0):
                uc_by_x_y0[i] = (u1_y0 + u2_y0 * sqrt_r_y0) / (1.0 + sqrt_r_y0)

            # ---------- 方案B: peak(start_idx) -> cross_idx ----------

            y_lower = y_line[start_idx:cross_idx + 1]
            ub_lower = ub_line[start_idx:cross_idx + 1]

            int_u_lower = np.trapz(ub_lower, y_lower)
            int_u2_lower = np.trapz(ub_lower**2, y_lower)
            int_u_upper = int_u_upper_y0
            int_u2_upper = int_u2_upper_y0

            int_alpha_lower = np.trapz(alpha_line[start_idx:cross_idx + 1], y_lower)
            int_alphaU_lower = np.trapz(alpha_line[start_idx:cross_idx + 1] * ub_lower, y_lower)
            int_alpha_upper = int_alpha_upper_y0
            int_alphaU_upper = int_alphaU_upper_y0

            u1 = int_u2_lower / int_u_lower if int_u_lower != 0 else np.nan
            u2 = int_u2_upper / int_u_upper if int_u_upper != 0 else np.nan
            alpha1 = int_alphaU_lower / int_alpha_lower if int_alpha_lower != 0 else np.nan
            alpha2 = int_alphaU_upper / int_alpha_upper if int_alpha_upper != 0 else np.nan
            alpha2 = min(max(alpha2, 0.0), 1.0)
            u1_by_x[i] = u1
            u2_by_x[i] = u2
            alpha1_by_x[i]= alpha1
            alpha2_by_x[i]= alpha2

            
            r = (alpha2 * self.rhoa + (1.0 - alpha2) * self.rhob) /(alpha1 * self.rhoa + (1.0 - alpha1) * self.rhob) 
            sqrt_r = np.sqrt(r) if r > 0 else np.nan
            if np.isfinite(sqrt_r) and np.isfinite(u1) and np.isfinite(u2):
                uc_by_x[i] = (u1 + u2 * sqrt_r) / (1.0 + sqrt_r)

        u1_2d_y0 = np.tile(u1_by_x_y0[:, None], (1, ny))
        u2_2d_y0 = np.tile(u2_by_x_y0[:, None], (1, ny))
        uc_2d_y0 = np.tile(uc_by_x_y0[:, None], (1, ny))
        u1_2d = np.tile(u1_by_x[:, None], (1, ny))
        u2_2d = np.tile(u2_by_x[:, None], (1, ny))
        uc_2d = np.tile(uc_by_x[:, None], (1, ny))

        # 3. 提取一维的物理坐标数组 (用于修正物理梯度的尺度)
        # x_2d 的形状是 (nx, ny)，每一列 x 值一样，所以取第 0 列就是 x_1d
        x_1d = x_2d[:, 0]
        y_1d = y_2d[0, :]
        
        # 4. 用修正后的物理间距，在平滑场上计算宏观 2D Q 准则！
        q_2d = self.compute_2D_Q_criterion(ub_2d, vb_2d, x_1d, y_1d)
        cores = self._find_vortex_cores(
            q_2d,
            x_plot_2d,
            x_2d,
            y_2d,
            alpha_a_2d,
            top_k=self.n_vortex_cores,
            percentile=self.q_core_percentile,
            min_separation=self.min_core_separation,
            sign_mode=self.q_core_sign,
            use_local_extrema=self.q_core_use_local_extrema,
            alpha_threshold=self.q_core_alpha_threshold,
        )

        core_csv = os.path.join(self.output_dir, f"vortex_cores_uc_t{time_dir}.csv")
        with open(core_csv, "w", encoding="ascii") as f:
            f.write(
                "core_id,i,j,x_headframe,x_physical,y,q_spanwise,"
                "u1_peak,u2_peak,uc_peak,alpha1_peak,alpha2_peak,"
                "u1_y0,u2_y0,uc_y0,alpha1_y0,alpha2_y0\n"
            )
            for k, core in enumerate(cores, start=1):
                u1_val = float(u1_2d[core["i"], core["j"]])
                u2_val = float(u2_2d[core["i"], core["j"]])
                uc_val = float(uc_2d[core["i"], core["j"]])
                alpha1_val = float(alpha1_by_x[core["i"]])
                alpha2_val = float(alpha2_by_x[core["i"]])
                u1_val_y0 = float(u1_2d_y0[core["i"], core["j"]])
                u2_val_y0 = float(u2_2d_y0[core["i"], core["j"]])
                uc_val_y0 = float(uc_2d_y0[core["i"], core["j"]])
                alpha1_val_y0 = float(alpha1_by_x_y0[core["i"]])
                alpha2_val_y0 = float(alpha2_by_x_y0[core["i"]])
                f.write(
                    f"{k},{core['i']},{core['j']},{core['x_headframe']:.9e},{core['x_physical']:.9e},"
                    f"{core['y']:.9e},{core['q']:.9e},{u1_val:.9e},{u2_val:.9e},{uc_val:.9e},"
                    f"{alpha1_val:.9e},{alpha2_val:.9e},{u1_val_y0:.9e},{u2_val_y0:.9e},{uc_val_y0:.9e},"
                    f"{alpha1_val_y0:.9e},{alpha2_val_y0:.9e}\n"
                )

        known_csv = self._write_known_points_u1u2_csv(
            time_dir=time_dir,
            x_2d=x_2d,
            y_2d=y_2d,
            x_headframe_2d=x_plot_2d,
            u1_2d=u1_2d,
            u2_2d=u2_2d,
            uc_2d=uc_2d,
        )

        if self.export_paraview:
            self._save_paraview_vtk(
                x_plot_2d,
                y_2d,
                ub_2d,
                scalar_name="Ub_x_spanwise",
                out_path=os.path.join(self.paraview_dir, f"Ub_x_spanwise_t{time_dir}.vtk"),
            )
            self._save_paraview_vtk(
                x_plot_2d,
                y_2d,
                alpha_a_2d,
                scalar_name="alpha_a_spanwise",
                out_path=os.path.join(self.paraview_dir, f"alpha_a_spanwise_t{time_dir}.vtk"),
            )
            self._save_paraview_vtk(
                x_plot_2d,
                y_2d,
                u1_2d,
                scalar_name="u1_lower_layer",
                out_path=os.path.join(self.paraview_dir, f"u1_lower_layer_t{time_dir}.vtk"),
            )
            self._save_paraview_vtk(
                x_plot_2d,
                y_2d,
                u2_2d,
                scalar_name="u2_upper_layer",
                out_path=os.path.join(self.paraview_dir, f"u2_upper_layer_t{time_dir}.vtk"),
            )
            self._save_paraview_vtk(
                x_plot_2d,
                y_2d,
                uc_2d,
                scalar_name="Uc_weighted",
                out_path=os.path.join(self.paraview_dir, f"Uc_weighted_t{time_dir}.vtk"),
            )
            self._save_paraview_vtk(
                x_plot_2d,
                y_2d,
                q_2d,
                scalar_name="Q_criterion_spanwise",
                out_path=os.path.join(self.paraview_dir, f"Q_criterion_spanwise_t{time_dir}.vtk"),
            )

        print(f"  finished t={time_dir}: computed u1/u2 for {np.sum(np.isfinite(u1_by_x))} x-columns")
        print(f"  detected {len(cores)} vortex cores, uc extracted to: {core_csv}")
        if known_csv is not None:
            print(f"  known-point u1/u2 extracted to: {known_csv}")

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.export_paraview:
            os.makedirs(self.paraview_dir, exist_ok=True)
        sort_idx, nx, ny, nz, x_2d, y_2d = self._build_sorted_mesh()

        for t in self.times:
            self.process_time_step(float(t), sort_idx, nx, ny, nz, x_2d, y_2d)


if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()