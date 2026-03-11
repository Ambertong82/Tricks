import numpy as np
import pandas as pd
import fluidfoam
import os

# ================= 1. 核心配置 =================
alpha_threshold = 1e-5 
rho_w = 1000.0          

def _depth_avg_weighted_trapz(field2d,  y_coords):
    """
    使用梯形积分法计算深度平均
    适用于不均匀y网格
    
    参数：
    field2d: 形状 (nx, ny) 的2D场
    y_coords: 形状 (ny,) 的y坐标（可不等距）
    
    返回：
    形状 (nx,) 的深度平均值
    """
    numerator = np.trapezoid(field2d , x=y_coords, axis=1)
    denominator = y_coords[-1] - y_coords[0]  # 总深度 (假设y_coords从浅到深排列)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)

def calc_gradient_tensor(field_data, coords):
    """
    通用张量梯度算子
    field_data: 可以是 [nx, ny] (标量) 也可以是 [n_comp, nx, ny] (向量)
    coords: [ux, uy]
    返回: 增加一个导数维度的张量
    """
    arr = np.asarray(field_data)
    if arr.ndim < 2:
        raise ValueError(f"field_data 维度不足，至少需要2维网格，当前 shape={arr.shape}")

    if len(coords) != 2:
        raise ValueError(f"当前仅支持二维梯度，coords 应为 [x, y]，当前长度={len(coords)}")

    grad_x, grad_y = np.gradient(arr, coords[0], coords[1], axis=(-2, -1), edge_order=1)
    return np.stack((grad_x, grad_y), axis=0)

def calc_second_order_gradient(field_data, coords):
    """
    计算二阶梯度，返回一个包含二阶导数的张量
    field_data: 可以是 [nx, ny] (标量) 也可以是 [n_comp, nx, ny] (向量)
    coords: [ux, uy]
    返回: 包含二阶导数的张量
    """
    arr = np.asarray(field_data)
    if arr.ndim < 2:
        raise ValueError(f"field_data 维度不足，至少需要2维网格，当前 shape={arr.shape}")

    if len(coords) != 2:
        raise ValueError(f"当前仅支持二维梯度，coords 应为 [x, y]，当前长度={len(coords)}")

    grad_xx = np.gradient(np.gradient(arr, coords[0], axis=-2), coords[0], axis=-2)
    grad_yy = np.gradient(np.gradient(arr, coords[1], axis=-1), coords[1], axis=-1)
    grad_xy = np.gradient(np.gradient(arr, coords[0], axis=-2), coords[1], axis=-1)

        # 返回Hessian矩阵 (2, 2, nx, ny)
    return np.stack([
        np.stack([grad_xx, grad_xy], axis=0),
        np.stack([grad_xy, grad_yy], axis=0)  # 使用对称性
    ], axis=0)

def extract_data(X_raw, Y_raw, Z_raw, sol, time_v):
    print(f"\n>>> 处理时间步: {time_v}")
    
    # 1. 数据读取 (保持不变)
    try:
        ua = fluidfoam.readvector(sol, str(time_v), "U.a")    
        ub = fluidfoam.readvector(sol, str(time_v), "U.b")    
        nut = fluidfoam.readscalar(sol, str(time_v), "nut.b") 
        k = fluidfoam.readscalar(sol, str(time_v), "k.b")     
        omega = fluidfoam.readscalar(sol, str(time_v), "omega.b")
        alpha_a = fluidfoam.readscalar(sol, str(time_v), "alpha.a")
        alpha_b = fluidfoam.readscalar(sol, str(time_v), "alpha.b")
        SUS = fluidfoam.readscalar(sol, str(time_v), "SUS")
        gamma = fluidfoam.readscalar(sol, str(time_v), "K")
    except Exception as e:
        print(f"读取失败: {e}"); return None

    # 2. 网格维度自动检测
    coordinate_x, coordinate_y, coordinate_z = np.unique(X_raw), np.unique(Y_raw), np.unique(Z_raw)
    nx, ny, nz = len(coordinate_x), len(coordinate_y), len(coordinate_z)
    print(f"网格规格: nx={nx}, ny={ny}, nz={nz}")

    # 3. 维度兼容转换函数 (使用 Order='F' 匹配 OpenFOAM)
    def to_3d(field): return field.reshape((nx, ny, nz), order='F')
    def v_to_3d(v_field): return v_field.reshape((3, nx, ny, nz), order='F')

    # 4. 动态平均逻辑 (核心改进)
    if nz > 1:
        print("检测到 3D 模拟，执行展向平均 (Spanwise Averaging)...")
        # 对 z 轴 (axis=2) 取平均
        Ubz_2d = np.mean(v_to_3d(ub), axis=3)  
        Uaz_2d = np.mean(v_to_3d(ua), axis=3)
        beta_2d = np.mean(to_3d(alpha_b), axis=2)
        nutz_2d = np.mean(to_3d(nut), axis=2)
        kz_2d = np.mean(to_3d(k), axis=2)
        omegaz_2d = np.mean(to_3d(omega), axis=2)
        alphaz_a_2d = np.mean(to_3d(alpha_a), axis=2)
        alphaz_b_2d = np.mean(to_3d(alpha_b), axis=2)
        SUS_2d = np.mean(to_3d(SUS), axis=2)
        gamma_2d = np.mean(to_3d(gamma), axis=2)
    else:
        print("检测到 2D 模拟，跳过展向平均，直接重构平面数据...")
        # 2D 情况下直接去掉 z 轴维度 [nx, ny, 1] -> [nx, ny]
        Ubz_2d = v_to_3d(ub)[:, :, :, 0]
        Uaz_2d = v_to_3d(ua)[:, :, :, 0]
        beta_2d = to_3d(alpha_b)[:, :, 0]
        nutz_2d = to_3d(nut)[:, :, 0]
        kz_2d = to_3d(k)[:, :, 0]
        omegaz_2d = to_3d(omega)[:, :, 0]
        alphaz_a_2d = to_3d(alpha_a)[:, :, 0]
        alphaz_b_2d = to_3d(alpha_b)[:, :, 0]
        SUS_2d = to_3d(SUS)[:, :, 0]
        gamma_2d = to_3d(gamma)[:, :, 0]

    # 5. 波前判断 (基于全域数据)
    # 在水相 alphaz_a_2d 存在的区域中找最大 x
    if not np.any(alphaz_a_2d > alpha_threshold):
        return None
    
    # 找到 head_x 在 ux 数组中的索引位置
    mask_x = np.any(alphaz_a_2d > alpha_threshold, axis=1) # 每一列是否有水
    valid_x_indices = np.where(mask_x)[0]
    if len(valid_x_indices) == 0: return None
    
    head_idx = valid_x_indices.max()
    head_x = coordinate_x[head_idx]
    print(f"波前位置: {head_x:.4f} m (索引: {head_idx})")
    print(f"Uaz_2d shape: {Uaz_2d.shape}")



    # ---  (2D 平面计算)-------------------------------
    # 1. 获取速度梯度张量 [i, j, x, y]
    # i=0,1 (dx, dy); j=0,1,2 (u, v, w)
    grad_U = calc_gradient_tensor(Ubz_2d, [coordinate_x, coordinate_y])
    grad_alpha = calc_gradient_tensor(alphaz_a_2d, [coordinate_x, coordinate_y])
    grad_beta = calc_gradient_tensor(alphaz_b_2d, [coordinate_x, coordinate_y])
    grad2_beta = calc_second_order_gradient(alphaz_b_2d, [coordinate_x, coordinate_y])
    print(f"gradUaz_2d shape: {grad_U.shape}")
    print(f"grad_alpha shape: {grad2_beta.shape}")
    print(f"ubz2d shape: {Ubz_2d.shape}")

    # 2. 构造对称应变率张量 S_ij = (du_i/dx_j + du_j/dx_i)
    # 注意：在 2D/3D 混合架构中，i 和 j 的范围需匹配 (通常取前两个分量)
    # 我们提取 2x2 的平面部分进行计算
    S_ij = grad_U[:2, :2] + np.transpose(grad_U[:2, :2], axes=(1, 0, 2, 3))

    # 3. 构造 Kronecker Delta 矩阵 [2, 2, nx, ny]
    delta_ij = np.eye(2)[:, :, np.newaxis, np.newaxis]

    # 4.1 组合成雷诺应力张量 tau_ij
    # kz_2d 和 nutz_2d 是之前算好的 [nx, ny] 场
    tau_ij = nutz_2d * S_ij - (2.0/3.0) * kz_2d * delta_ij
    print(f"tau_ij shape: {tau_ij.shape}")
    

    G = rho_w * alphaz_b_2d *np.einsum('ijxy,ijxy->xy', tau_ij, grad_U[:2, :2])  # 生产项

    # 4.2 计算密度梯度项
    grad_outer_density= np.einsum('ixy,jxy->ijxy', grad_beta, grad_beta)  # 密度梯度的外积
    density_gradient1 = np.einsum('ijxy,ijxy->xy', grad_outer_density, tau_ij)  # 密度梯度*雷诺应力
    density_gradient2 = np.einsum('ijxy,ijxy->xy', grad2_beta, tau_ij)  # 二阶密度梯度*雷诺应力                
    density_gradient = nutz_2d*1e-6*rho_w/SUS_2d*(density_gradient1 - density_gradient2)  # 总的密度梯度项            
   
    # 4.3 dissipation
    epsilon = 0.09*kz_2d*omegaz_2d 
    dissipation = -alphaz_b_2d*rho_w*epsilon

    #4.4 drag1
    veldiff = -Uaz_2d[:2] + Ubz_2d[:2]  # 速度差 (2, nx, ny)
    print(f"veldiff shape: {veldiff.shape}")
    veldotgradbeta = np.einsum('ixy,ixy->xy', grad_alpha, veldiff)
    drag1 = gamma_2d*nutz_2d/SUS_2d/beta_2d*veldotgradbeta

    #4.5 drag2
    drag2 = 2*gamma_2d*(1/np.sqrt(SUS_2d)-1)*alphaz_a_2d*kz_2d

    #4.6 drag3
    coeff3 = 2*gamma_2d*(1/np.sqrt(SUS_2d)-1)*alphaz_b_2d*kz_2d*nutz_2d/SUS_2d/omegaz_2d
    laplacian_beta = np.einsum('ijxy,ijxy->xy', grad2_beta, delta_ij)  # Δβ = ∂²β/∂x² + ∂²β/∂y²
    drag3 = coeff3*laplacian_beta

    # 7. 深度平均与 Head Mask 提取
    # G_depth_avg = _depth_avg_weighted_trapz(G_2d, coordinate_y)
    G_depth_avg = _depth_avg_weighted_trapz(G, coordinate_y)
    densityGrad_depth_avg = _depth_avg_weighted_trapz(density_gradient, coordinate_y)
    dissipation_depth_avg = _depth_avg_weighted_trapz(dissipation, coordinate_y)
    drag1_depth_avg = _depth_avg_weighted_trapz(drag1, coordinate_y)    
    drag2_depth_avg = _depth_avg_weighted_trapz(drag2, coordinate_y)
    drag3_depth_avg = _depth_avg_weighted_trapz(drag3, coordinate_y)
    
    # 提取从 0 到 head_x 的结果
    final_x = coordinate_x[:head_idx+1]
    final_G = G_depth_avg[:head_idx+1]
    final_densityGrad = densityGrad_depth_avg[:head_idx+1]
    final_dissipation = dissipation_depth_avg[:head_idx+1]
    final_drag1 = drag1_depth_avg[:head_idx+1]
    final_drag2 = drag2_depth_avg[:head_idx+1]
    final_drag3 = drag3_depth_avg[:head_idx+1]

    return {'time': float(time_v), 
            'x': final_x.tolist(), 
            'G_mean': final_G.tolist(),
            'densityGrad_mean': final_densityGrad.tolist(),
            'dissipation_mean': final_dissipation.tolist(),
            'drag1_mean': final_drag1.tolist(),
            'drag2_mean': final_drag2.tolist(),
            'drag3_mean': final_drag3.tolist()
            
            
            }

# (save_data 和 main 函数保持与上一版一致)

# ================= 4. 保存与运行 =================
def save_data(results, output_path, prefix):
    if not results: return
    # 由于每个时间步的 head_x 不同，导致 x 的长度不同
    # 建议每个时间步保存一个独立文件，或者存为长表格式
    for r in results:
        df = pd.DataFrame({
            'x': r['x'],
            'G_production': r['G_mean'],
            'density_gradient': r['densityGrad_mean'],
            'dissipation': r['dissipation_mean'],
            'drag1': r['drag1_mean'],
            'drag2': r['drag2_mean'],
            'drag3': r['drag3_mean']
        })
        filename = f"{prefix}_t{r['time']:.2f}.csv"
        df.to_csv(os.path.join(output_path, filename), index=False)
        print(f"已保存: {filename} (数据点数: {len(r['x'])})")

def main():
    # 修改为你的实际路径
    sol = '/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/NEW/case230307_5' 
    output = '/home/amber/postpro/selecting_variant'
    if not os.path.exists(output): os.makedirs(output)
    
    X, Y, Z = fluidfoam.readmesh(sol)
    times = [15] # 示例时间步
    
    all_results = []
    for t in times:
        res = extract_data(X, Y, Z, sol, t)
        if res:
            all_results.append(res)
    
    save_data(all_results, output, 'TKE_Budget_Spatial')

if __name__ == "__main__":
    main()