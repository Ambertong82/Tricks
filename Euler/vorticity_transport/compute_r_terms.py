def compute_r_terms(data, gradUb, finite_diff, vector_calc, rho_f):
    """计算扩散和粘性相关项。"""
    # R1: 扩散项 beta * nuEff * laplacian(vorticity)
    lap_vort = finite_diff.compute_laplacian_simple(data.X, data.Y, data.Z, data.vorticityUb)
    R1 = data.beta * lap_vort * rho_f * data.nuEffb

    # R2: 粘性项 cross(grad(beta*nuEff), laplacian(Ub))
    betanuEff = data.beta * data.nuEffb
    grad_betanuEff = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, betanuEff)
    lap_Ub = finite_diff.compute_laplacian_simple(data.X, data.Y, data.Z, data.Ub)
    R2 = vector_calc.cross_product(grad_betanuEff, lap_Ub) * rho_f

    # R3: curl[(grad(beta*nuEff) · grad(Ub))]
    # 这里 dotGradient 结果应为向量 (3, nx, ny, nz)。
    dotGradient = vector_calc.tensor_vector_contraction(gradUb, grad_betanuEff) * rho_f
    R3 = finite_diff.compute_vorticity_simple(data.X, data.Y, data.Z, dotGradient)

    # R4: baroclinic effect = grad(beta*nuEff) x grad(div(Ub))
    # 期望 divUb 形状: (nx, ny, nz)。
    # 若上游返回 (1, nx, ny, nz)，此处压缩掉长度为 1 的伪向量轴，
    # 避免后续 compute_gradient_simple 与广播错位。
    divUb = vector_calc.divergence(gradUb)
    if divUb.ndim == 4 and divUb.shape[0] == 1:
        divUb = divUb[0]
    gradDiv = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, divUb)
    R4 = vector_calc.cross_product(grad_betanuEff, gradDiv) * rho_f

    # R5: curl[(Hessian(beta*nuEff) · Ub)]
    # second_rhobeta 是 (3, 3, nx, ny, nz)，Ub 是 (3, nx, ny, nz)。
    second_rhobeta = finite_diff.compute_second_derivative_scalar(data.X, data.Y, data.Z, betanuEff)
    grad_gUb = vector_calc.tensor_vector_contraction(second_rhobeta, data.Ub) * rho_f
    R5 = finite_diff.compute_vorticity_simple(data.X, data.Y, data.Z, grad_gUb)

    # R6: 拖曳力导致的涡量差项。
    vorticity_diff = data.vorticityUa - data.vorticityUb
    R6 = data.alpha_A * vorticity_diff * data.gamma

    # R7: 拖曳力导致的速度差项。
    velocity_diff = data.Ub - data.Ua
    grad_Alpha = finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.alpha_A)
    R7 = vector_calc.cross_product(grad_Alpha, velocity_diff) * data.gamma

    return {"R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5, "R6": R6, "R7": R7}
