#!/usr/bin/env python3
"""
initial_setting.py  (standalone, no mesh)

Plot the depth-resolved sediment volume fraction alpha(y) and the contact
pressure pa(y), using exactly the same logic that init_alpha.py applies to the
OpenFOAM cell centres -- only here the y-column is defined directly instead of
being read from 0/Cy, so the figure can be produced without a mesh.

Sign convention follows init_alpha.py: y is the physical vertical coordinate,
the bed sits at y < 0 and the water column at y > 0.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- physics / constitutive params (sync with init_alpha.py) ----
rhos = 2650.0
rhof = 1000.0
g    = 9.81
K    = 1.0e+09
kai  = 5.39
min_alpha = 0.54
max_alpha = 0.625

# ---- interface smoothing (original tanh: alpha = A*0.5*(1+tanh((z0-y)/w))) ----
z0 = 0.00065  # bed-surface reference level (y of the tanh centre) 0.00065  -0.0015
w  = 0.0005    # smoothing half-width


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
    """Sediment fraction at height y. The tanh envelope decays to 0 on both
    sides of y = z0, so alpha is continuous across the interface."""
    d = z0 - y
    env = 0.5 * (1.0 + np.tanh(d / w))
    return alpha_at_depth(d) * env


def pa_at_y(y):
    """Contact pressure pa = (rhos-rhof)*g*d (non-negative), same envelope;
    exactly the hydrostatic overburden that alpha_at_depth was solved for."""
    d = z0 - y
    env = 0.5 * (1.0 + np.tanh(d / w))
    return (rhos - rhof) * g * max(d, 0.0) * env


# ---- sampling column: y defined directly, no mesh needed ----
y_bottom = -0.05
y_top    = 0.01
dy       = 0.0001

yp = np.arange(y_bottom, y_top + 0.5 * dy, dy)

alpha = np.array([alpha_at_y(y) for y in yp])
pa = np.array([pa_at_y(y) for y in yp])

# ---- statistics over the bed (y < 0), same report as init_alpha ----
bed = yp < 0.0
print(f"alpha range = {alpha.min():.4f} .. {alpha.max():.4f}"
      f"   bed mean = {alpha[bed].mean():.4f}")
print(f"pa    range = {pa.min():.4f} .. {pa.max():.1f} Pa"
      f"   bed mean = {pa[bed].mean():.1f} Pa")

# ---- spot queries, same y values init_alpha prints for the real mesh ----
print("\n=== queried y values ===")
print(f"{'y (m)':>11} {'alpha':>8} {'pa (Pa)':>14}")
print("-" * 35)
for yq in (-0.05, -0.03, -0.01, -0.003, 0.0, 0.001, 0.003):
    print(f"{yq:11.6f} {alpha_at_y(yq):8.4f} {pa_at_y(yq):14.4e}")

# ---- visualization ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

ax[0].plot(alpha, yp, 'b-', label='Computed Alpha')
ax[0].axvline(x=0.6, color='r', linestyle='--', label='Target Mean (0.6)')
ax[0].axhline(y=0.0, color='k', linestyle=':', lw=0.8)
ax[0].set_xlabel('Solid Volumetric Fraction')
ax[0].set_ylabel('y (m)')
ax[0].set_title('Distribution of Solid Fraction')
ax[0].legend(loc='best')
ax[0].grid(True)

ax[1].plot(pa, yp, 'r-', label='Contact Pressure')
ax[1].axhline(y=0.0, color='k', linestyle=':', lw=0.8)
ax[1].set_xlabel('Enduring Contact Pressure (Pa)')
ax[1].set_title('Pressure Profile')
ax[1].legend(loc='best')
ax[1].grid(True)

plt.tight_layout()
plt.show()
