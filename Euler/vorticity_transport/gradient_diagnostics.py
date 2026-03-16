import numpy as np
from itertools import permutations, product


def diagnose_gradient_component(
    grad_calc: np.ndarray,
    grad_ref: np.ndarray,
    diff: np.ndarray,
    data,
    i: int,
    j: int,
    label: str,
    top_k: int = 10,
):
    """定位单个梯度分量误差最大的区域，并判断是否由边界或参考值过小导致。"""
    term_calc = grad_calc[i, j]
    term_ref = grad_ref[i, j]
    term_diff = diff[i, j]

    abs_diff = np.abs(term_diff)
    rel_diff = abs_diff / (np.abs(term_ref) + 1e-12)
    flat_abs = abs_diff.ravel()
    #abs_diff.ravel() 是把多维数组（如 (nx, ny, nz)）拉平成一维视图。
    # 目的通常是为了方便：
    # - argmax 找全局最大误差位置（再用 unravel_index 还原回三维索引）
    # - argpartition 取 top-k 最大误差点


    if flat_abs.size == 0:
        print(f"\n  [{label}] 无可用数据用于诊断")
        return

    max_flat_idx = np.argmax(flat_abs)
    max_idx = np.unravel_index(max_flat_idx, abs_diff.shape)
    x_idx, y_idx, z_idx = max_idx

    print(f"\n  Detailed diagnosis for {label}:")
    print(f"    Ref norm   = {np.linalg.norm(term_ref):.4e}")
    print(f"    Calc norm  = {np.linalg.norm(term_calc):.4e}")
    print(f"    Diff norm  = {np.linalg.norm(term_diff):.4e}")
    print(f"    Mean|ref|  = {np.mean(np.abs(term_ref)):.4e}")
    print(f"    Mean|diff| = {np.mean(abs_diff):.4e}")
    print(f"    Max|diff|  = {abs_diff[max_idx]:.4e}")
    print(f"    Max relerr = {rel_diff[max_idx]:.4e}")
    print(
        f"    Worst point index = (ix={x_idx}, iy={y_idx}, iz={z_idx}), "
        f"coord = ({data.X[max_idx]:.6f}, {data.Y[max_idx]:.6f}, {data.Z[max_idx]:.6f})"
    )
    print(f"    OpenFOAM value = {term_ref[max_idx]:.4e}")
    print(f"    Calculated val = {term_calc[max_idx]:.4e}")
    print(f"    Difference     = {term_diff[max_idx]:.4e}")

    y_axis = data.Y[0, :, 0]
    if y_axis.size >= 2:
        first_dy = y_axis[1] - y_axis[0]
        print(
            f"    y-axis check: y[0]={y_axis[0]:.6e}, y[1]={y_axis[1]:.6e}, "
            f"(y[1]-y[0])={first_dy:.6e}"
        )

    boundary_masks = {
        'x_min': np.zeros_like(abs_diff, dtype=bool),
        'x_max': np.zeros_like(abs_diff, dtype=bool),
        'y_min': np.zeros_like(abs_diff, dtype=bool),
        'y_max': np.zeros_like(abs_diff, dtype=bool),
        'z_min': np.zeros_like(abs_diff, dtype=bool),
        'z_max': np.zeros_like(abs_diff, dtype=bool),
    }
    boundary_masks['x_min'][0, :, :] = True
    boundary_masks['x_max'][-1, :, :] = True
    boundary_masks['y_min'][:, 0, :] = True
    boundary_masks['y_max'][:, -1, :] = True
    boundary_masks['z_min'][:, :, 0] = True
    boundary_masks['z_max'][:, :, -1] = True

    print("    Boundary mean |diff|:")
    for name, mask in boundary_masks.items():
        if np.any(mask):
            print(f"      - {name}: {np.mean(abs_diff[mask]):.4e}")

    if min(abs_diff.shape) > 2:
        interior_mask = np.zeros_like(abs_diff, dtype=bool)
        interior_mask[1:-1, 1:-1, 1:-1] = True
        print(f"      - interior: {np.mean(abs_diff[interior_mask]):.4e}")

    small_ref_mask = np.abs(term_ref) < 1e-8
    if np.any(small_ref_mask):
        print(
            f"    Fraction with |ref| < 1e-8: "
            f"{np.mean(small_ref_mask) * 100:.2f}%"
        )
        print(
            f"    Mean|diff| where |ref|<1e-8: "
            f"{np.mean(abs_diff[small_ref_mask]):.4e}"
        )

    top_k = min(top_k, flat_abs.size)
    top_indices = np.argpartition(flat_abs, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(flat_abs[top_indices])[::-1]]

    print(f"    Top {top_k} worst points by |diff|:")
    for rank, flat_idx in enumerate(top_indices, start=1):
        idx = np.unravel_index(flat_idx, abs_diff.shape)
        print(
            f"      {rank:02d}. idx={idx}, "
            f"coord=({data.X[idx]:.6f}, {data.Y[idx]:.6f}, {data.Z[idx]:.6f}), "
            f"ref={term_ref[idx]:.4e}, calc={term_calc[idx]:.4e}, "
            f"|diff|={abs_diff[idx]:.4e}, rel={rel_diff[idx]:.4e}"
        )


def compare_gradients(data, finite_diff):
    """对比自定义梯度计算与 OpenFOAM 原始数据，分析边界误差。"""
    grad_calc = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.Ub)

    grad_of = data.gradUb.reshape(3, 3, *data.X.shape, order='C')
    grad_of_jac = np.swapaxes(grad_of, 0, 1) # OpenFOAM 输出的梯度是 dU_i/dx_j 的形式，需要转置为 dU_j/dx_i 以匹配自定义计算的格式

    diff = grad_calc - grad_of_jac
    print(f"\n{'='*20} Gradient Verification (t={data.time}) {'='*20}")

    component_errors = {}

    boundary_mask = np.zeros_like(diff[0, 0], dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask

    for i, comp_i in enumerate(['x', 'y', 'z']):
        for j, comp_j in enumerate(['x', 'y', 'z']):
            term_of = grad_of_jac[i, j]
            term_diff = diff[i, j]

            ref_val = np.linalg.norm(term_of)
            l2_err = np.linalg.norm(term_diff) / (ref_val + 1e-10)
            component_errors[(i, j)] = l2_err
            print(f"  dU{comp_i}/d{comp_j}: Relative L2 Error = {l2_err:.4e}")

            if np.any(boundary_mask):
                b_ref = np.linalg.norm(term_of[boundary_mask])
                b_err = np.linalg.norm(term_diff[boundary_mask]) / (b_ref + 1e-10)
            else:
                b_err = np.nan

            if np.any(interior_mask):
                in_ref = np.linalg.norm(term_of[interior_mask])
                in_err = np.linalg.norm(term_diff[interior_mask]) / (in_ref + 1e-10)
            else:
                in_err = np.nan

            print(f"    boundary L2 = {b_err:.4e}, interior L2 = {in_err:.4e}")

    print(f"\n  Boundary Error Analysis (Mean Absolute Difference):")
    y_min_err = np.mean(np.abs(diff[:, :, :, 0, :]))
    print(f"    - Bottom Boundary (y_min): {y_min_err:.4e}")
    z_edge_err = np.mean(np.abs(diff[:, :, :, :, [0, -1]]))
    print(f"    - Side Boundaries (z_edges): {z_edge_err:.4e}")
    inner_err = np.mean(np.abs(diff[:, :, 1:-1, 1:-1, 1:-1]))
    print(f"    - Interior Points:         {inner_err:.4e}")

    worst_component = max(component_errors, key=component_errors.get)
    worst_i, worst_j = worst_component
    labels = ['x', 'y', 'z']
    diagnose_gradient_component(
        grad_calc,
        grad_of_jac,
        diff,
        data,
        worst_i,
        worst_j,
        f"dU{labels[worst_i]}/d{labels[worst_j]}",
    )

    compare_vorticity(data, finite_diff)
    print(f"{'='*65}\n")


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)


def _find_best_vorticity_mapping(vort_calc: np.ndarray, vort_of: np.ndarray):
    """尝试所有分量排列+符号组合，找与 OpenFOAM 涡量最匹配的顺序。"""
    best = {
        'err': np.inf,
        'perm': (0, 1, 2),
        'signs': (1, 1, 1),
        'mapped': vort_calc,
    }

    for perm in permutations((0, 1, 2)):
        permuted = vort_calc[list(perm)]
        for signs in product((1.0, -1.0), repeat=3):
            signs_arr = np.array(signs)[:, None, None, None]
            candidate = permuted * signs_arr
            err = _relative_l2(candidate, vort_of)
            if err < best['err']:
                best = {
                    'err': err,
                    'perm': perm,
                    'signs': signs,
                    'mapped': candidate,
                }
    return best


def compare_vorticity(data, finite_diff):
    """对比 OpenFOAM 涡量与本程序涡量，诊断分量顺序与符号是否一致。"""
    vort_calc = finite_diff.compute_vorticity_simple(data.X, data.Y, data.Z, data.Ub)
    vort_of = data.vorticityUb

    if vort_calc.shape != vort_of.shape:
        print("\n[Vorticity Verification] shape mismatch:")
        print(f"  calc shape={vort_calc.shape}, of shape={vort_of.shape}")
        return

    print(f"\n{'='*20} Vorticity Verification (t={data.time}) {'='*20}")

    comp_labels = ['x', 'y', 'z']
    for i, lbl in enumerate(comp_labels):
        l2 = _relative_l2(vort_calc[i], vort_of[i])
        print(f"  omega_{lbl} direct Relative L2 = {l2:.4e}")

    mag_calc = np.linalg.norm(vort_calc, axis=0)
    mag_of = np.linalg.norm(vort_of, axis=0)
    mag_l2 = _relative_l2(mag_calc, mag_of)
    print(f"  |omega| Relative L2 = {mag_l2:.4e}")

    best = _find_best_vorticity_mapping(vort_calc, vort_of)
    perm = best['perm']
    signs = best['signs']
    print(
        "  Best mapping (calc -> of): "
        f"perm={perm}, signs={signs}, Relative L2={best['err']:.4e}"
    )
    print("  Mapping detail:")
    for of_idx, calc_idx in enumerate(perm):
        sign_char = '+' if signs[of_idx] > 0 else '-'
        print(f"    omega_of_{comp_labels[of_idx]} ~= {sign_char} omega_calc_{comp_labels[calc_idx]}")

    mapped = best['mapped']
    for i, lbl in enumerate(comp_labels):
        l2 = _relative_l2(mapped[i], vort_of[i])
        print(f"  omega_{lbl} mapped Relative L2 = {l2:.4e}")

    boundary_mask = np.zeros_like(mag_calc, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    interior_mask = ~boundary_mask

    if np.any(boundary_mask):
        b_mag_l2 = _relative_l2(mag_calc[boundary_mask], mag_of[boundary_mask])
    else:
        b_mag_l2 = np.nan
    if np.any(interior_mask):
        i_mag_l2 = _relative_l2(mag_calc[interior_mask], mag_of[interior_mask])
    else:
        i_mag_l2 = np.nan

    print(f"  |omega| boundary L2 = {b_mag_l2:.4e}, interior L2 = {i_mag_l2:.4e}")
