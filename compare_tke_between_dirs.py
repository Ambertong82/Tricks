#!/usr/bin/env python3
"""Plot TKE curves from two TKE-budget output directories separately.

Usage:
    python compare_tke_between_dirs.py \
        --dir1 /path/to/first/output \
        --dir2 /path/to/second/output \
        [--outdir /path/to/save/plots]

By default the script searches for CSV files named like
`TKE_Budget_SpanAvg_VAvg_t{time:.2f}.csv` inside `curve_t*` folders
under the provided base directories. It extracts the TKE column
prefer `TKE_avg`) and `x_dime`. You can also overlay selected times in one
figure using the nondimensional time scale `t*0.85`.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIR1 = "/home/amber/postpro/TKE_budget/tc3d_d23_0327_1"
DEFAULT_DIR2 = "/home/amber/postpro/TKE_budget/tc3d_d23_0428_5"


def find_curve_csvs(base_dir: str) -> Dict[float, str]:
    """Return map time->csvpath for CSVs under curve_t* subfolders."""
    pattern = os.path.join(base_dir, "curve_t*", "Drag_plus_Dissipation_over_G_t*.csv")
    files = glob.glob(pattern)
    out: Dict[float, str] = {}
    for f in files:
        m = re.search(r"t([0-9]+(?:\.[0-9]+)?)\.csv$", f)
        if not m:
            # try a more permissive search
            m2 = re.search(r"_t([0-9]+(?:\.[0-9]+)?)", f)
            if m2:
                t = float(m2.group(1))
            else:
                continue
        else:
            t = float(m.group(1))
        out[t] = f
    return out


def read_tke_from_csv(csv_path: str) -> Optional[Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    # Prefer exact column name, otherwise try heuristics.
    col = None
    candidates = ["TKE_avg", "TKE", "k_avg", "k", "k.b"]
    for c in candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        # case-insensitive search for 'tke' or leading 'k'
        for c in df.columns:
            nc = c.lower()
            if "tke" in nc or nc.startswith("k"):
                col = c
                break

    if col is None:
        return None

    if "x_dime" not in df.columns and "x" in df.columns:
        xcol = "x"
    else:
        xcol = "x_dime"

    if xcol not in df.columns:
        return None

    return {"x": df[xcol].to_numpy(dtype=float), "y": df[col].to_numpy(dtype=float), "colname": col}


def make_single_plot(x, y, label: str, outpath: str, time_v: float,color=None ):
    fig_size = (20, 3)
    curve_lw = 2.0
    x_dime_max = 4.0
    dimtime = time_v * 0.85

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(x, y, linewidth=curve_lw, color=color, label=label)
    ax.set_title(rf"TKE at $t^*={dimtime:.2f}$", fontsize=22)
    ax.set_xlabel(r"$(x_f-x)/H$", fontsize=20)
    ax.set_xlim(x_dime_max, 0.0)
    ax.set_ylabel(r"$k^{*}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    # 2. 获取左上角那个 '1e-X' 文本对象，并强制修改它的字体大小
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(16)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def make_time_overlay_plot(curves, outpath: str, title: str):
    fig_size = (20, 3)
    curve_lw = 2.0
    x_dime_max = 4.0
    
    

    fig, ax = plt.subplots(figsize=fig_size)
    for x, y, label in curves:
        ax.plot(x, y, linewidth=curve_lw, color='blue', label=label)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"$(x_f-x)/H$", fontsize=12)
    ax.set_xlim(x_dime_max, 0.0)
    ax.set_ylabel(r"$k^{*}$", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir1", default=DEFAULT_DIR1)
    p.add_argument("--dir2", default=DEFAULT_DIR2)
    p.add_argument("--outdir", default=None)
    p.add_argument("--overlay-times", nargs="*", type=float, default=[10.0, 25.0], help="Times to overlay on the same figure")
    args = p.parse_args()

    dir1 = args.dir1
    dir2 = args.dir2
    outdir = "/home/amber/postpro/TKE_budget/comparison_plots" if args.outdir is None else args.outdir
    os.makedirs(outdir, exist_ok=True)

    map1 = find_curve_csvs(dir1)
    map2 = find_curve_csvs(dir2)

    times = sorted(set(map1.keys()) | set(map2.keys()))
    if not times:
        print("No curve CSV files found in either directory.")
        return

    print(f"Found times in dir1: {sorted(map1.keys())}")
    print(f"Found times in dir2: {sorted(map2.keys())}")

    overlay_times = [t for t in args.overlay_times if t in times]
    if len(overlay_times) >= 2:
        dir1_curves = []
        dir2_curves = []
        for t in overlay_times:
            if t in map1:
                d1 = read_tke_from_csv(map1[t])
                if d1 is not None:
                    nondim_t = t * 0.85
                    dir1_curves.append((d1["x"], d1["y"], f"without correction, t*={nondim_t:.2f}"))

            if t in map2:
                d2 = read_tke_from_csv(map2[t])
                if d2 is not None:
                    nondim_t = t * 0.85
                    dir2_curves.append((d2["x"], d2["y"], f"with correction, t*={nondim_t:.2f}"))

        if dir1_curves:
            overlay_outpath1 = os.path.join(outdir, "dir1_TKE_overlay_t10_t25.png")
            overlay_title1 = " TKE curves "
            make_time_overlay_plot(dir1_curves, overlay_outpath1, overlay_title1)
            print(f"Saved {overlay_outpath1}")

        if dir2_curves:
            overlay_outpath2 = os.path.join(outdir, "dir2_TKE_overlay_t10_t25.png")
            overlay_title2 = "TKE curves "
            make_time_overlay_plot(dir2_curves, overlay_outpath2, overlay_title2)
            print(f"Saved {overlay_outpath2}")

    for t in times:
        

        p1 = map1.get(t)
        p2 = map2.get(t)
        if p1 is None and p2 is None:
            continue

        d1 = read_tke_from_csv(p1) if p1 else None
        d2 = read_tke_from_csv(p2) if p2 else None

        if d1 is None and d2 is None:
            print(f"No readable TKE column for time {t}")
            continue

        if d1 is not None:
            label1 = "without correction"
            outpath1 = os.path.join(outdir, f"dir1_TKE_t{t:.2f}.png")
            make_single_plot(d1["x"], d1["y"], label1, outpath1, t)
            print(f"Saved {outpath1}")

        if d2 is not None:
            label2 = "with correction"
            outpath2 = os.path.join(outdir, f"dir2_TKE_t{t:.2f}.png")
            make_single_plot(d2["x"], d2["y"], label2, outpath2, t,color='orange')
            print(f"Saved {outpath2}")

        if d1 is None and d2 is None:
            print(f"No readable TKE column for time {t}")


if __name__ == "__main__":
    main()
