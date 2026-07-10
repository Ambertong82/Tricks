"""
垂向平均 alpha 得到 xz 平面云图
====================================
逻辑:
  1. 读 3D alpha.a, U.b → 重构为结构化网格 (nx, ny, nz)
  2. 对每个 (x, z) 垂向柱:
     a. 取 Ubx(y) 垂直剖面 → 找零速点 (velocity zero-crossing) → H(x,z)
     b. 在 [y_lower, H(x,z)] 内筛选 alpha > 1e-5 的区域 → 垂向平均
  3. 结果: (nx, nz) 的 xz 平面场 → contourf 云图

坐标约定 (与 TKE_budget_ofcal.py 一致):
  - x (nx): 流向
  - y (ny): 垂向 (高度)
  - z (nz): 展向

注: H 不再基于展向平均的 Ubx 计算, 而是每个 (x,z) 用自己位置处的 Ubx(y) 垂直剖面,
    得到各自独立的 H(x,z). 这在保留展向维度时更合理.
"""

import os
from typing import Dict, Optional

import fluidfoam
import numpy as np

class AlphaVerticalAvgXZ:
    """垂向平均 alpha.a, 输出 xz 平面云图 (流向 × 展向)."""

    def __init__(self):
        # --- 算例路径 ---
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        self.output_dir = "/home/amber/postpro/TKE_budget/tc3d_d23_0428_4/alpha_vavg_xz"
        self.times = [15,25, 30,35]

        # --- 物理参数 ---
        self.U = 0.255
        self.H0 = 0.3          # 参考高度, 用于无量纲化

        # --- alpha 筛选 ---
        self.alpha_vavg_threshold = 1e-5    # 垂向平均时仅统计 alpha > 此值的层

        # --- H 零速搜索参数 ---
        self.y_lower = 0.001   # 从该高度之上开始找零速点

    # ------------------------------------------------------------------
    # 网格工具
    # ------------------------------------------------------------------
    @staticmethod
    def _build_grid_cache(X_raw, Y_raw, Z_raw) -> Dict[str, np.ndarray]:
        x_axis = np.unique(X_raw)
        y_axis = np.unique(Y_raw)
        z_axis = np.unique(Z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))
        return {
            "sort_idx": sort_idx,
            "nx": nx, "ny": ny, "nz": nz,
            "x_axis_3d": x_axis,
            "y_axis_3d": y_axis,
            "z_axis_3d": z_axis,
        }

    @staticmethod
    def _reshape_sorted(field, sort_idx, nx, ny, nz) -> np.ndarray:
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    # ------------------------------------------------------------------
    # H 计算: 对单个 Ubx(y) 垂直剖面找零速点 → 动量厚度
    # 与 TKE_budget_ofcal._vertical_average_to_zerocity_zero 一致
    # ------------------------------------------------------------------
    @staticmethod
    def _velocity_height_from_profile(u_profile: np.ndarray,
                                       y_coords: np.ndarray,
                                       y_lower: float = 0.001) -> float:
        """从一条 Ubx(y) 垂直剖面计算动量厚度 H = (∫u dy)² / ∫(u²) dy."""
        valid = np.isfinite(u_profile) & np.isfinite(y_coords)
        y_valid = y_coords[valid]
        u_valid = u_profile[valid]

        if y_valid.size < 2:
            return 0.0

        # 找零速点: Ubx 从正 → 负/零
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
        u_sel = u_valid[active_mask]

        if y_sel.size < 2:
            return 0.0

        # 动量厚度 H = (∫u dy)² / ∫(u²) dy
        if hasattr(np, "trapezoid"):
            int_u = float(np.trapezoid(u_sel, x=y_sel))
            int_u2 = float(np.trapezoid(u_sel ** 2, x=y_sel))
        else:
            int_u = float(np.trapz(u_sel, x=y_sel))
            int_u2 = float(np.trapz(u_sel ** 2, x=y_sel))

        return int_u ** 2 / int_u2 if abs(int_u2) > 1e-20 else 0.0

    # ------------------------------------------------------------------
    # 垂向平均: 对每个 (x, z) 柱, 沿 y 筛选 alpha > threshold 做平均
    # ------------------------------------------------------------------
    def _vertical_average_alpha_xz(self, alpha_3d: np.ndarray,
                                    ubx_3d: np.ndarray,
                                    y_axis: np.ndarray,
                                    ) -> np.ndarray:
        """对每个 (x, z) 位置:
        1. 从 Ubx(y) 垂直剖面计算 H(x,z)
        2. 在 [y_lower, H(x,z)] 内筛选 alpha > threshold
        3. 梯形积分平均

        Parameters
        ----------
        alpha_3d : (nx, ny, nz) — 原始 3D alpha.a
        ubx_3d   : (nx, ny, nz) — Ubx 的 x 分量 3D 场
        y_axis   : (ny,)        — 垂向坐标

        Returns
        -------
        alpha_vavg : (nx, nz) — 垂向平均后的 xz 平面场
        H_out      : (nx, nz) — 每个 (x,z) 的 H 值 (用于诊断)
        """
        nx, ny, nz = alpha_3d.shape

        alpha_vavg = np.full((nx, nz), np.nan, dtype=float)
        H_out = np.full((nx, nz), np.nan, dtype=float)

        for i in range(nx):
            for k in range(nz):
                # --- 算 H(x,z) ---
                ubx_profile = ubx_3d[i, :, k]          # Ubx(y) at (x[i], z[k])
                h_local = self._velocity_height_from_profile(
                    ubx_profile, y_axis, self.y_lower
                )
                H_out[i, k] = h_local

                if h_local <= 1e-12:
                    continue

                # --- 垂向平均 alpha ---
                profile = alpha_3d[i, :, k]             # alpha(y) at (x[i], z[k])

                y_mask = ((y_axis >= self.y_lower)
                          & (y_axis <= h_local)
                          & (profile > self.alpha_vavg_threshold)
                          & np.isfinite(profile))

                if np.count_nonzero(y_mask) < 2:
                    continue

                y_sel = y_axis[y_mask]
                a_sel = profile[y_mask]

                if hasattr(np, "trapezoid"):
                    integral = float(np.trapezoid(a_sel, x=y_sel))
                else:
                    integral = float(np.trapz(a_sel, x=y_sel))

                dy = float(y_sel[-1] - y_sel[0])
                alpha_vavg[i, k] = integral / dy if dy > 1e-12 else float(a_sel[0])

        return alpha_vavg, H_out

    # ------------------------------------------------------------------
    # 定位 current head (front) 位置
    # ------------------------------------------------------------------
    def _locate_head_index(self, alpha_a_2d: np.ndarray,
                           alpha_threshold: float = 1e-3) -> Optional[int]:
        mask_x = np.any(alpha_a_2d > alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if len(valid_x) == 0:
            return None
        return int(valid_x.max())

    # ------------------------------------------------------------------
    # VTK 输出 (2D 结构化网格: 流向 x 展向)
    # ------------------------------------------------------------------
    @staticmethod
    def _write_structured_grid_vtk(
        out_path: str,
        x_axis: np.ndarray,
        z_axis: np.ndarray,
        scalars: Dict[str, np.ndarray],
    ) -> None:
        """写 2D 结构化网格 VTK (nx, nz, 1)."""
        nx, nz = x_axis.size, z_axis.size

        with open(out_path, "w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Vertically-averaged alpha on XZ plane\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {nz} 1\n")
            f.write(f"POINTS {nx * nz} float\n")

            for k in range(nz):
                for i in range(nx):
                    f.write(f"{float(x_axis[i]):.9e} {float(z_axis[k]):.9e} 0.000000000e+00\n")

            f.write(f"POINT_DATA {nx * nz}\n")
            for name, field in scalars.items():
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for k in range(nz):
                    for i in range(nx):
                        val = float(field[i, k])
                        if not np.isfinite(val):
                            val = 0.0
                        f.write(f"{val:.9e}\n")

    def _save_xz_vtk(self, x_axis: np.ndarray,
                     z_axis: np.ndarray,
                     alpha_xz: np.ndarray,
                     H_xz: np.ndarray,
                     time_v: float,
                     head_idx: int,
                     head_x: float):
        """输出 head 前区域 alpha_vavg 和 H 到 VTK (无量纲坐标)."""
        os.makedirs(self.output_dir, exist_ok=True)
        time_tag = f"{time_v:.2f}"

        # 无量纲坐标
        x_seg = x_axis[:head_idx + 1]
        x_dime = (head_x - x_seg) / self.H0   # (x_f - x)/H0
        z_dime = z_axis / self.H0

        # 截取 head 之前的场
        alpha_seg = np.nan_to_num(alpha_xz[:head_idx + 1, :], nan=0.0)
        H_seg = np.nan_to_num(H_xz[:head_idx + 1, :], nan=0.0)

        # H 也无量纲化
        scalars = {
            "alpha_vavg": alpha_seg,
            "H_over_H0": H_seg / self.H0,
        }

        out_path = os.path.join(self.output_dir,
                                f"alpha_vavg_xz_t{time_tag}.vtk")
        self._write_structured_grid_vtk(out_path, x_dime, z_dime, scalars)
        print(f"  Saved VTK: {out_path}")

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def process_time_step(self, grid, time_v: float):
        time_dir = f"{float(time_v):g}"
        time_tag = f"{time_v:.2f}"
        nondim_time = time_v * 0.85
        time_label = rf"$t*={nondim_time:.2f}$"
        print(f"\n>>> Processing time: {time_label}")

        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]
        x_axis = grid["x_axis_3d"]
        y_axis = grid["y_axis_3d"]
        z_axis = grid["z_axis_3d"]

        print(f"  Mesh: nx={nx}, ny={ny}, nz={nz}")
        print(f"  x range: [{x_axis[0]:.4f}, {x_axis[-1]:.4f}]")
        print(f"  y range: [{y_axis[0]:.6f}, {y_axis[-1]:.6f}]")
        print(f"  z range: [{z_axis[0]:.6f}, {z_axis[-1]:.6f}]")

        # ---- 读 alpha.a ----
        try:
            alpha_raw = fluidfoam.readscalar(self.sol, time_dir, "alpha.a")
        except Exception as exc:
            print(f"  Read alpha.a failed: {exc}")
            return
        alpha_3d = self._reshape_sorted(alpha_raw, sort_idx, nx, ny, nz)

        # ---- 读 U.b ----
        try:
            ub_raw = fluidfoam.readvector(self.sol, time_dir, "U.b")
        except Exception as exc:
            print(f"  Read U.b failed: {exc}")
            return
        Ub = self._reshape_sorted(ub_raw, sort_idx, nx, ny, nz)
        ubx_3d = Ub[0]    # (nx, ny, nz) — Ubx 全 3D 场

        # 找 head 位置 (用展向平均 alpha)
        alpha_2d = np.mean(alpha_3d, axis=2)    # (nx, ny)
        head_idx = self._locate_head_index(alpha_2d)
        if head_idx is None:
            print(f"  No head found at {time_tag}")
            return
        head_x = float(x_axis[head_idx])
        print(f"  Head: x={head_x:.4f} (idx={head_idx})")

        # ---- 垂向平均 alpha → xz 平面 ----
        # 每个 (x,z) 用自己的 Ubx(y) 剖面算 H, 再垂向平均
        alpha_xz, H_xz = self._vertical_average_alpha_xz(
            alpha_3d, ubx_3d, y_axis
        )

        # 诊断: H 的展向变化幅度
        H_valid = H_xz[:head_idx + 1, :][np.isfinite(H_xz[:head_idx + 1, :])]
        if H_valid.size > 0:
            print(f"  H range: [{H_valid.min():.4f}, {H_valid.max():.4f}] m")
            print(f"  H mean:  {H_valid.mean():.4f} m")

        # ---- 输出 VTK ----
        self._save_xz_vtk(x_axis, z_axis, alpha_xz, H_xz, time_v,
                          head_idx, head_x)

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        X_raw, Y_raw, Z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(X_raw, Y_raw, Z_raw)

        for t in self.times:
            self.process_time_step(grid, float(t))


if __name__ == "__main__":
    AlphaVerticalAvgXZ().run_analysis()
