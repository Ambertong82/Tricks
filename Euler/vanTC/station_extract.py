"""按头部后方固定距离抽取已算好的逐时刻 CSV（不重跑主程序）
==============================================================
relative_vel.py 每个时刻输出一份 band_speed_bed_exchange_t*.csv，行是网格列 x。
本脚本只读这些 CSV，把同一目标距离（0.2 / 0.5 / 1.0 m）处的所有后处理结果
沿时间收集起来，每个距离输出一份 CSV：一行一个时刻，一列一个输出量。

  - 不读场、不重算；以后新增的后处理只要在本文件里加函数
  - 输出目录与配置直接取自 relative_vel.py（import 不会执行它的 main）
  - 各时刻 CSV 的列不一致时取并集，缺的列写 nan
    （例如旧的 t41 文件没有 muI_thickness_m）

输出（relative_vel.output_dir 下）:
  station_distance_0.2m.csv
  station_distance_0.5m.csv
  station_distance_1m.csv
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relative_vel as rv  # 只借用 output_dir / tname；import 不会跑主程序


DISTANCES_M = [0.2, 0.5, 1.0]
CSV_PATTERN = "band_speed_bed_exchange_t*.csv"
TIME_COLUMN = "Time"


def time_from_filename(path):
    """从 band_speed_bed_exchange_t17.csv 里取出 17。"""
    match = re.search(r"_t([0-9.]+)\.csv$", os.path.basename(path))
    if match is None:
        raise ValueError(f"文件名里没有时间: {path}")
    return float(match.group(1))


def read_csv_table(path):
    """返回 (列名列表, 数据矩阵)；数据矩阵按原始列顺序。"""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return header, data


def sample_at_distance(header, data, target):
    """在 distance_behind_head_m = target 处线性插值，返回 {列名: 值}。"""
    column = {name: i for i, name in enumerate(header)}
    distance = data[:, column["distance_behind_head_m"]]
    samples = {}
    for name, j in column.items():
        values = data[:, j]
        keep = np.isfinite(distance) & np.isfinite(values)
        if keep.sum() < 2:
            samples[name] = np.nan
            continue
        d = distance[keep]
        v = values[keep]
        order = np.argsort(d)
        samples[name] = float(np.interp(target, d[order], v[order],
                                        left=np.nan, right=np.nan))
    return samples


def main():
    paths = sorted(glob.glob(os.path.join(rv.output_dir, CSV_PATTERN)),
                   key=time_from_filename)
    if not paths:
        raise SystemExit(f"{rv.output_dir} 里没有 {CSV_PATTERN}")

    # 读一遍所有时刻的表；列名取并集，旧文件缺的列后面写 nan
    tables = []
    times = []
    columns = []
    for path in paths:
        header, data = read_csv_table(path)
        tables.append((path, header, data))
        times.append(time_from_filename(path))
        for name in header:
            if name not in columns:
                columns.append(name)
    if TIME_COLUMN in columns:
        columns.remove(TIME_COLUMN)

    for target in DISTANCES_M:
        rows = np.full((len(tables), len(columns)), np.nan)
        for i, (_path, header, data) in enumerate(tables):
            samples = sample_at_distance(header, data, target)
            for k, name in enumerate(columns):
                if name in samples:
                    rows[i, k] = samples[name]

        table = np.column_stack([np.asarray(times), rows])
        out_csv = os.path.join(rv.output_dir,
                               f"station_distance_{target:g}m.csv")
        np.savetxt(out_csv, table, delimiter=",", fmt="%.10g", comments="",
                   header=",".join([TIME_COLUMN] + columns))
        print(f"distance={target:g} m: {len(tables)} 个时刻 -> {out_csv}")

    for path, header, _data in tables:
        missing = [name for name in columns if name not in header]
        if missing:
            print(f"  {os.path.basename(path)} 缺少: {', '.join(missing)} (写 nan)")


if __name__ == "__main__":
    main()
