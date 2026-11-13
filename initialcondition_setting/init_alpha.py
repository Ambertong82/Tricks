#!/usr/bin/env python3
"""
init_alpha.py  (in the OpenFOAM case directory)

Read the cell-centre y-coordinate field `0/Cy` (written by
    postProcess -func writeCellCentres -time 0
which lists the cell centres in the same order as the internalField),
and overwrite `0/alpha.a` with a depth-resolved sediment volume fraction
that follows the hydrostatic packing profile alpha_at_depth() solved in
initialcondition_setting/initial_setting.py, smoothed at the bed surface
with the original tanh envelope (centre z0, width w).

Physics constants must stay in sync with initial_setting.py.
"""

import numpy as np
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent

# ---- physics / constitutive params (sync with initial_setting.py) ----
rhos = 2650.0
rhof = 1000.0
g    = 9.81
K    = 1.0e+09
kai  = 5.39
min_alpha = 0.54
max_alpha = 0.625

# ---- interface smoothing (original tanh: alpha = A*0.5*(1+tanh((z0-y)/w))) ----
z0 = 0.00065   # bed-surface reference level (y of the tanh centre)
w  = 0.0005      # smoothing half-width

# ----------------------------------------------------------------------------

def pse(alpha):
    return K * ((alpha - min_alpha) ** kai) * (
        1.0 + np.sin((alpha - min_alpha) * np.pi / (max_alpha - min_alpha) - np.pi / 2.0)
    )


def alpha_at_depth(d, tol=0.01):
    """Solve pse(alpha) = (rhos-rhof)*g*d for alpha in [min_alpha, max_alpha]."""
    p0 = (rhos - rhof) * g * d
    if p0 <= 0:
        return min_alpha
    # linearised initial guess
    t1 = min_alpha + (p0 * (max_alpha - min_alpha) / (2.0 * K)) ** (1.0 / (1.0 + kai))
    if abs(p0 - pse(t1)) / p0 < tol:
        return t1
    # bisection over the physical range
    a1, a2 = min_alpha, max_alpha
    am = 0.5 * (a1 + a2)
    while abs(p0 - pse(am)) / p0 >= tol:
        if pse(am) < p0:
            a1 = am
        else:
            a2 = am
        am = 0.5 * (a1 + a2)
    return am


def alpha_at_y(y):
    """Sediment fraction at cell centre y. The tanh envelope decays to 0 on
    BOTH sides of y = z0 (no hard cut), so alpha is continuous across the
    interface with a smooth shoulder into the water, like the original tanh."""
    d = z0 - y
    env = 0.5 * (1.0 + np.tanh(d / w))
    return alpha_at_depth(d) * env


def pa_at_y(y):
    """Contact pressure pa = (rhos-rhof)*g*d (non-negative), same envelope;
    exactly the hydrostatic overburden that alpha_at_depth was solved for."""
    d = z0 - y
    env = 0.5 * (1.0 + np.tanh(d / w))
    return (rhos - rhof) * g * max(d, 0.0) * env


def read_cy(path):
    """Return the list of cell-centre y values from an OpenFOAM scalarField."""
    txt = path.read_text()
    key = "nonuniform List<scalar>"
    i = txt.index(key)
    j = txt.index("(", i)
    k = txt.index("\n)", j)  # list closes on its own line: ')' then ';'
    return [float(x) for x in txt[j + 1:k].split()]


def read_field_skeleton(path):
    """Split an OpenFOAM volScalarField into (prefix, suffix) around internalField."""
    txt = path.read_text()
    i = txt.index("internalField")
    j = txt.index("boundaryField")
    return txt[:i], txt[j:]


def write_field(name, vals):
    """Overwrite 0/<name>: keep FoamFile header + boundaryField, swap internalField."""
    dst = CASE / "0" / name
    prefix, suffix = read_field_skeleton(dst)
    body = (f"internalField   nonuniform List<scalar> \n{len(vals)}\n(\n"
            + "\n".join(f"{v:.6g}" for v in vals)
            + "\n)\n;\n\n")
    dst.write_text(prefix + body + suffix)
    print(f"[init_alpha] wrote {dst}")


def main():
    write = "--write" in sys.argv
    cy = read_cy(CASE / "0" / "Cy")
    n = len(cy)
    if n != 264000:
        print(f"[init_alpha] warning: {n} cells read (expect 264000)")
    ys = np.array(cy)

    # compute once per distinct y, then scatter back (fields depend on y only)
    uniq, inv = np.unique(np.round(ys, 7), return_inverse=True)
    alpha = np.array([alpha_at_y(float(yu)) for yu in uniq])[inv]
    pa = np.array([pa_at_y(float(yu)) for yu in uniq])[inv]

    # statistics
    bed = alpha[ys < 0.0]
    print(f"[init_alpha] cells = {n}   bed cells (y<0) = {bed.size}")
    print(f"[init_alpha] alpha range = {alpha.min():.4f} .. {alpha.max():.4f}"
          f"   bed mean = {bed.mean():.4f}")
    print(f"[init_alpha] pa     range = {pa.min():.4f} .. {pa.max():.1f} Pa"
          f"   bed mean = {pa[ys < 0.0].mean():.1f} Pa")
    for yq in (-0.05, -0.03, -0.01, -0.003, 0.0, 0.001, 0.003):
        kk = int(np.argmin(np.abs(ys - yq)))
        print(f"[init_alpha]   y = {ys[kk]:11.6f}  alpha = {alpha[kk]:8.4f}"
              f"  pa = {pa[kk]:14.4e}")

    if not write:
        print("[init_alpha] dry run (no file written). Add --write to write 0/alpha.a and 0/pa")
        return

    write_field("alpha.a", alpha)
    #write_field("pa", pa)


if __name__ == "__main__":
    main()