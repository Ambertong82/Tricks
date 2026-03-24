import numpy as np
def compute_l_terms(data, gradUb, finite_diff, vector_calc, rho_f, prev_omega_mom=None, dt=None):
    """计算涡量输运方程左端项。"""
    # L0: 使用 FiniteDifference 计算的速度梯度 (3, 3, nx, ny, nz) 用于验证项。
    L0 = gradUb

    # L_ddt_omega_mom: d/dt [curl(rho*beta*Ub)]，使用后向差分。
    momentum = data.Ub * data.beta * rho_f
    omega_mom = finite_diff.compute_vorticity_simple(data.X, data.Y, data.Z, momentum)
    if prev_omega_mom is None or dt is None or dt <= 0:
        L_ddt_omega_mom = np.zeros_like(omega_mom)
    else:
        L_ddt_omega_mom = (omega_mom - prev_omega_mom) / dt

    # L1: 密度梯度跟动能梯度的叉乘。
    # grad(0.5*|U|^2) 这里沿用原脚本写法，用 |U| * grad|U| 近似表达。
    gradmagUb = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.magUb)
    gradke = gradmagUb * data.magUb[None, ...]
    L1 = vector_calc.cross_product(data.gradbeta, gradke) * rho_f

    # L2: Lamb 向量与密度梯度的叉乘。
    print(f"Max gradUb: {np.max(np.abs(gradUb))}")
    print(f"Max vorticity: {np.max(np.abs(data.vorticityUb))}")
    # 如果 gradUb > 100 或者 vorticity > 100，说明你的空间导数计算由于网格太细而炸了
    lamb = vector_calc.cross_product(data.Ub, data.vorticityUb)
    L2 = vector_calc.cross_product(lamb, data.gradbeta) * rho_f

    # L3: 涡量拉伸项 (Tensor contraction)。
    vort_2d = np.zeros_like(data.vorticityUb)
    # vort_2d[2] = data.vorticityUb[2] # 只保留 omega_z
    # L3 = vector_calc.tensor_vector_contraction(gradUb, vort_2d) * data.beta * rho_f
    L3 = vector_calc.tensor_vector_contraction(gradUb, data.vorticityUb) * data.beta * rho_f

    # L4: rho * beta * (Ua · ∇)omega。
    grad_vortUb = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.vorticityUb)
    L4 = vector_calc.tensor_vector_contraction(grad_vortUb, data.Ua) * data.beta * rho_f

    # L5: 压缩项 (Vorticity * Divergence)。
    divUb = vector_calc.divergence(gradUb)
    L5 = data.beta * data.vorticityUb * divUb[None, ...] * rho_f

    return {
        "L0": L0,
        "L_ddt_omega_mom": L_ddt_omega_mom,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "L5": L5,
    }, omega_mom
