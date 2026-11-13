"""头部区域平均浓度 alpha_mean(t)
====================================
对每个时间步:
  1. 定位头部 x 位置 head_x (alpha.a > alpha_threshold 的最大 x 坐标,
     三维时先在展向上平均再找, 与 front_speed_2davg.py / alpha_vertical_average_xy.py 一致)
  2. 取头部后方一段长度 L = 0.3 * H0 的窗口: x in [head_x - L, head_x]
  3. 垂向统一从 y = 0 起积分, 上边界逐 (x, z) 遍历: 每个柱各自找 alpha 首次降到
     alpha_threshold 以下的 y 作为该柱的界面高度 y_top(x, z) (每个柱的高度不同)
  4. 对整个区域做体积分 (alpha_threshold 只用于定上边界, 积分本身不过滤):
         alpha_mean = ∫∫∫ alpha dV / ∫∫∫ dV
     分母是该区域随 y_top(x, z) 起伏的真实体积, 不是长方体体积
  5. 二维算例 (nz=1) 与三维算例 (nz>1, 展向全部计入) 走同一套代码, 无需改动:
     - 二维: 展向平均退化为自身, 区域就是 (流向 x 垂向) 的一个矩形
     - 三维: 展向所有网格点都进入积分 (等效于把展向宽度加进积分区)

坐标约定 (与 TKE_budget_ofcal.py / front_speed_2davg.py 一致):
  x (nx): 流向   y (ny): 垂向   z (nz): 展向

输出: 每个时间步一行的 CSV (Time, head_x, x_start, x_end, y_top_mean, alpha_mean, ...)
"""

import os
from typing import Dict, Optional

import fluidfoam
import numpy as np


class HeadRegionMeanAlpha:
    def __init__(self):
        # --- 算例路径 ---
        # 三维:
        self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_4"
        # 二维:
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/FIne_particle9/2d/case090604_2"
        # self.sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/2D/case230604_2"

        self.output_dir = "/home/amber/postpro/head_region_alpha"

        # --- 时间序列 ---
        self.times = np.arange(1, 40, 0.5)

        # --- 场名与阈值 ---
        self.alpha_field = "alpha.a"
        self.alpha_threshold = 1e-5

        # --- 平均区几何 ---
        self.H0 = 0.3                    # 参考高度 (m), 固定值
        self.region_fraction = 0.3       # 窗口长度 = region_fraction * H0
        self.region_length = self.region_fraction * self.H0

        # 垂向积分下限: 统一从 y = 0 起算 (None 表示改用网格底部)
        self.y_bottom: Optional[float] = 0.0

    # ------------------------------------------------------------------
    # 网格工具
    # ------------------------------------------------------------------
    @staticmethod
    def _time_to_dir_name(time_v: float) -> str:
        return f"{float(time_v):g}"

    @staticmethod
    def _build_grid_cache(x_raw: np.ndarray,
                          y_raw: np.ndarray,
                          z_raw: np.ndarray) -> Dict[str, np.ndarray]:
        x_axis = np.unique(x_raw)
        y_axis = np.unique(y_raw)
        z_axis = np.unique(z_raw)
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        # 按 x -> y -> z 排序, 使场数据能 reshape 成 (nx, ny, nz)
        sort_idx = np.lexsort((z_raw, y_raw, x_raw))
        return {
            "sort_idx": sort_idx,
            "nx": nx, "ny": ny, "nz": nz,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
        }

    @staticmethod
    def _reshape_sorted(field: np.ndarray, sort_idx: np.ndarray,
                        nx: int, ny: int, nz: int) -> np.ndarray:
        if field.ndim == 1:
            return field[sort_idx].reshape((nx, ny, nz), order="C")
        return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")

    # ------------------------------------------------------------------
    # 定位头部: alpha > threshold 的最大 x 坐标
    # ------------------------------------------------------------------
    def _locate_head_index(self, alpha_2d: np.ndarray) -> Optional[int]:
        mask_x = np.any(alpha_2d > self.alpha_threshold, axis=1)
        valid_x = np.where(mask_x)[0]
        if valid_x.size == 0:
            return None
        return int(valid_x.max())

    # ------------------------------------------------------------------
    # 单列界面高度: alpha 从底部往上首次降到 threshold 以下的 y
    # ------------------------------------------------------------------
    def _y_bottom_value(self, y_axis: np.ndarray) -> float:
        """垂向积分下限 (self.y_bottom = None 时取网格底部)."""
        return float(y_axis[0]) if self.y_bottom is None else float(self.y_bottom)

    def _column_top_height(self, profile: np.ndarray,
                           y_axis: np.ndarray) -> Optional[float]:
        """返回该 (x, z) 柱的界面高度 y_top.

        以最上面一个 alpha > threshold 的网格点为准, 再往上找第一个
        alpha < threshold 的点作为界面; 若整列都没有超过阈值的点则返回 None.
        这样即使床面附近有低于阈值的点也不会把界面高度截断在底部.
        """
        y_bottom = self._y_bottom_value(y_axis)
        idx = np.where((y_axis >= y_bottom) & np.isfinite(profile))[0]
        if idx.size == 0:
            return None

        prof = profile[idx]
        above = prof > self.alpha_threshold
        if not above.any():
            return None

        last = int(np.max(np.where(above)[0]))
        after = np.where(prof[last:] < self.alpha_threshold)[0]
        if after.size:
            return float(y_axis[idx[last + int(after[0])]])
        return float(y_axis[idx[-1]])

    # ------------------------------------------------------------------
    # 单时间步: 窗口内平均浓度
    # ------------------------------------------------------------------
    def _region_mean_at_time(self, grid: Dict[str, np.ndarray],
                             time_v: float) -> Optional[dict]:
        time_dir = self._time_to_dir_name(time_v)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        sort_idx = grid["sort_idx"]
        x_axis = grid["x_axis"]
        y_axis = grid["y_axis"]
        z_axis = grid["z_axis"]

        try:
            alpha_raw = fluidfoam.readscalar(self.sol, time_dir, self.alpha_field)
        except Exception as exc:
            print(f"  Read {self.alpha_field} failed at t={time_dir}: {exc}")
            return None

        alpha_3d = self._reshape_sorted(alpha_raw, sort_idx, nx, ny, nz)

        # 头部位置: 三维用展向平均, 二维 (nz=1) 退化为自身
        alpha_2d = np.mean(alpha_3d, axis=2)
        head_ix = self._locate_head_index(alpha_2d)
        if head_ix is None:
            print(f"  t={time_dir}: no head found "
                  f"(alpha <= {self.alpha_threshold:g})")
            return None

        head_x = float(x_axis[head_ix])
        x_end = head_x
        x_start = head_x - self.region_length
        # 窗口起始列: 第一个 x >= x_start 的网格点
        i_start = int(np.searchsorted(x_axis, x_start, side="left"))
        i_start = min(i_start, head_ix)

        # 逐列 (x, z) 取界面高度 y_top(x, z), 然后做体积积分
        #   alpha_mean = Σ(alpha * dV) / Σ(dV),   dV = dx * dy * dz
        # 积分区内不再按 alpha_threshold 过滤 (阈值只用于定 y_top)
        y_bottom = self._y_bottom_value(y_axis)
        # 变间距网格下的单元尺寸
        dx_cell = np.gradient(x_axis)
        dy_cell = np.gradient(y_axis)
        dz_cell = np.gradient(z_axis)

        top_heights = []
        sum_alpha_dv = 0.0   # 分子: ∫∫∫ alpha dV
        sum_dv = 0.0         # 分母: ∫∫∫ dV (纯体积)
        n_cells = 0
        for i in range(i_start, head_ix + 1):
            dxi = float(dx_cell[i])
            for k in range(nz):
                profile = alpha_3d[i, :, k]
                y_top = self._column_top_height(profile, y_axis)
                if y_top is None:
                    continue
                top_heights.append(y_top)

                sel = ((y_axis >= y_bottom)
                       & (y_axis <= y_top)
                       & np.isfinite(profile))
                if not np.any(sel):
                    continue

                dV = dxi * float(dz_cell[k]) * dy_cell[sel]
                sum_alpha_dv += float(np.sum(profile[sel] * dV))
                sum_dv += float(np.sum(dV))
                n_cells += int(np.count_nonzero(sel))

        n_columns = int((head_ix - i_start + 1) * nz)
        if sum_dv <= 0.0:
            print(f"  t={time_dir}: head at x={head_x:.4f} but empty "
                  f"integration region inside the window")
            return None

        return {
            "time": float(time_v),
            "head_x": head_x,
            "x_start": x_start,
            "x_end": x_end,
            "y_top_mean": float(np.mean(top_heights)),
            "alpha_mean": sum_alpha_dv / sum_dv,   # 体积分 / 体积
            "volume": sum_dv,
            "n_cells": n_cells,
            "n_columns": n_columns,
        }

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        case_name = os.path.basename(os.path.normpath(self.sol))
        output_csv = os.path.join(
            self.output_dir,
            f"head_region_alpha_{case_name}_{self.alpha_threshold:g}.csv",
        )

        x_raw, y_raw, z_raw = fluidfoam.readmesh(self.sol)
        grid = self._build_grid_cache(x_raw, y_raw, z_raw)
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        dim_label = "2D" if nz == 1 else "3D"
        print(f"Case: {self.sol}")
        print(f"Mesh ({dim_label}): nx={nx}, ny={ny}, nz={nz}")

        rows = []
        for t in self.times:
            row = self._region_mean_at_time(grid, float(t))
            if row is None:
                continue
            rows.append(row)
            print(f"t={row['time']:g}: head_x={row['head_x']:.4f}, "
                  f"window x=[{row['x_start']:.4f}, {row['x_end']:.4f}], "
                  f"y_top_mean={row['y_top_mean']:.4f}, "
                  f"alpha_mean={row['alpha_mean']:.6f} "
                  f"(V={row['volume']:.6e} m^3, n_cells={row['n_cells']})")

        if not rows:
            print("No valid time step. Nothing written.")
            return

        header = ["Time", "head_x", "x_start", "x_end", "y_top_mean",
                  "alpha_mean", "volume", "n_cells", "n_columns"]
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(f"{r['time']:.10g},{r['head_x']:.10g},"
                        f"{r['x_start']:.10g},{r['x_end']:.10g},"
                        f"{r['y_top_mean']:.10g},{r['alpha_mean']:.10g},"
                        f"{r['volume']:.10g},"
                        f"{r['n_cells']},{r['n_columns']}\n")

        print(f"\nSaved {len(rows)} time steps to: {output_csv}")


if __name__ == "__main__":
    HeadRegionMeanAlpha().run()
