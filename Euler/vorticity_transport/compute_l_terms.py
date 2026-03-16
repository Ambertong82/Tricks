def compute_l_terms(data, gradUb, finite_diff, vector_calc, rho_f):
    """计算涡量输运方程左端项。"""
    # L0: 使用 FiniteDifference 计算的速度梯度 (3, 3, nx, ny, nz) 用于验证项。
    L0 = gradUb

    # L1: 密度梯度跟动能梯度的叉乘。
    # grad(0.5*|U|^2) 这里沿用原脚本写法，用 |U| * grad|U| 近似表达。
    gradmagUb = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.magUb)
    gradke = gradmagUb * data.magUb[None, ...]
    L1 = vector_calc.cross_product(data.gradbeta, gradke) * rho_f

    # L2: Lamb 向量与密度梯度的叉乘。
    lamb = vector_calc.cross_product(data.Ub, data.vorticityUb)
    L2 = vector_calc.cross_product(lamb, data.gradbeta) * rho_f

    # L3: 涡量拉伸项 (Tensor contraction)。
    L3 = vector_calc.tensor_vector_contraction(gradUb, data.vorticityUb) * data.beta * rho_f

    # L4: rho * beta * (Ua · ∇)omega。
    grad_vortUb = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.vorticityUb)
    L4 = vector_calc.tensor_vector_contraction(grad_vortUb, data.Ua) * data.beta * rho_f

    # L5: 压缩项 (Vorticity * Divergence)。
    divUb = vector_calc.divergence(gradUb)
    L5 = data.beta * data.vorticityUb * divUb[None, ...] * rho_f

    return {"L0": L0, "L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}
