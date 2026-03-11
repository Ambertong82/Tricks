import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
# from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# ============================================================================
# 数学运算工具类 - 规范化向量和微分运算
# ============================================================================

class VectorCalculus:
    """封装所有向量和张量运算操作 - 支持结构化 4D 数组 (3, nx, ny, nz)"""
    
    @staticmethod
    def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """批量叉乘计算: a x b"""
        # 假设输入维度为 (3, nx, ny, nz)
        cross_x = a[1] * b[2] - a[2] * b[1]
        cross_y = a[2] * b[0] - a[0] * b[2]
        cross_z = a[0] * b[1] - a[1] * b[0]
        return np.stack([cross_x, cross_y, cross_z], axis=0)
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """批量点积计算: a · b"""
        return np.einsum('i...,i...->...', a, b)
    
    @staticmethod
    def dot_product_vector_gradient(grad_f: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """向量与张量的点积 (grad_f · Tensor)"""

        if tensor.shape[0] == 9:
            shape = tensor.shape[1:]
            tensor = tensor.reshape(3, 3, *shape, order='C')
        
        # tensor[i, j] * grad_f[j] -> result[i]
        return np.einsum('ijn...,jn...->in...', tensor, grad_f)

    @staticmethod  
    def tensor_vector_contraction(tensor: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """张量与向量的缩并 (Tensor · vector)"""
        if tensor.shape[0] == 9:
            shape = tensor.shape[1:]
            # 根据 OpenFOAM 的排列更正: xx yx zx; xy yy zy...
            tensor = tensor.reshape(3, 3, *shape, order='C')
        # tensor[i, j] * vector[j] -> result[i]
        return np.einsum('ijn...,jn...->in...', tensor, vector)
        
    @staticmethod
    def divergence(grad_U: np.ndarray) -> np.ndarray:
        """计算散度 trace(grad_U)"""
        if grad_U.shape[0] == 9:
            return grad_U[0] + grad_U[4] + grad_U[8]
        return grad_U[0, 0] + grad_U[1, 1] + grad_U[2, 2]

# ============================================================================
# 微分算子类 - 规范化数值微分计算 (支持非均匀网格)
# ============================================================================

class FiniteDifference:
    """结构化网格上的数值微分计算 - 精简规范版"""
    
    @staticmethod
    def _get_axes(cx, cy, cz):
        return cx[:, 0, 0], cy[0, :, 0], cz[0, 0, :]

    @staticmethod
    def compute_gradient_simple(cx, cy, cz, f):
        """计算梯度 ∇f。支持标量 (nx,ny,nz) -> (3,nx,ny,nz) 或 向量 (3,nx,ny,nz) -> (3,3,nx,ny,nz)"""
        ax, ay, az = FiniteDifference._get_axes(cx, cy, cz)
        grid_shape = cx.shape # (nx, ny, nz)
        
        # 判断是标量场还是向量场
        if f.shape == grid_shape:
            # 标量情况
            grad = np.zeros((3, *grid_shape))
            grad[0] = np.gradient(f, ax, axis=0)
            grad[1] = np.gradient(f, ay, axis=1)
            grad[2] = np.gradient(f, az, axis=2)
            return grad
        elif f.ndim == 4 and f.shape[0] == 3:
            # 向量情况 (3, nx, ny, nz)
            grad = np.zeros((3, 3, *grid_shape))
            for i in range(3):
                ##  这里第一个索引 i 选的是速度分量 U_i，第二个索引选的是求导方向，所以它就是：- grad[i,j] = ∂U_i / ∂x_j（Jacobian 约定）
                grad[i, 0] = np.gradient(f[i], ax, axis=0)
                grad[i, 1] = np.gradient(f[i], ay, axis=1)
                grad[i, 2] = np.gradient(f[i], az, axis=2)
            return grad
        else:
            raise ValueError(f"Unsupported field shape {f.shape} for grid {grid_shape}")

    @staticmethod
    def compute_vorticity_simple(cx, cy, cz, U):
        """计算涡量 ω = ∇ × U"""
        ax, ay, az = FiniteDifference._get_axes(cx, cy, cz)
        Ux, Uy, Uz = U
        vort = np.zeros_like(U)
        vort[0] = np.gradient(Uz, ay, axis=1) - np.gradient(Uy, az, axis=2)
        vort[1] = np.gradient(Ux, az, axis=2) - np.gradient(Uz, ax, axis=0)
        vort[2] = np.gradient(Uy, ax, axis=0) - np.gradient(Ux, ay, axis=1)
        return vort

    @staticmethod
    def compute_second_derivative_simple(cx, cy, cz, f):
        """计算各个方向的二阶偏导数 (∂²f/∂x², ∂²f/∂y², ∂²f/∂z²)"""
        ax, ay, az = FiniteDifference._get_axes(cx, cy, cz)
        
        def _second_deriv(field):
            d2_dx2 = np.gradient(np.gradient(field, ax, axis=0), ax, axis=0)
            d2_dy2 = np.gradient(np.gradient(field, ay, axis=1), ay, axis=1)
            d2_dz2 = np.gradient(np.gradient(field, az, axis=2), az, axis=2)
            return np.stack([d2_dx2, d2_dy2, d2_dz2], axis=0)
            
        if f.ndim == 3: # 标量 (nx, ny, nz)
            return _second_deriv(f)
        else: # 向量 (3, nx, ny, nz)
            # 返回维度: (向量分量i, 导数方向j, nx, ny, nz) -> ∂²Ui/∂xj²
            return np.stack([_second_deriv(f[i]) for i in range(3)], axis=0)

    @staticmethod
    def compute_laplacian_simple(cx, cy, cz, f):
        """计算拉普拉斯算子 ∇²f = Σ(∂²f/∂xi²)"""
        # 利用二阶导数的结果进行求和，体现包含关系
        d2f = FiniteDifference.compute_second_derivative_simple(cx, cy, cz, f)
        
        # 对于标量 f: d2f 维度为 (3, nx, ny, nz), 对 axis=0 (导数方向) 求和
        # 对于矢量 f: d2f 维度为 (3, 3, nx, ny, nz), 对 axis=1 (导数方向) 求和
        axis = 1 if f.ndim == 4 else 0
        return np.sum(d2f, axis=axis)

@dataclass
class TimeStepData:
    """单个时间步的数据容器"""
    time: float
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray # (nx, ny, nz)
    alpha_A: np.ndarray
    beta: np.ndarray # 体积分数
    Ua: np.ndarray
    Ub: np.ndarray # 速度场 (3, nx, ny, nz)
    vorticityUa: np.ndarray # 涡量
    vorticityUb: np.ndarray # 涡量
    gradUb: np.ndarray # 速度梯度 (9, nx, ny, nz)
    gradbeta: np.ndarray # 密度梯度
    nuEffb: np.ndarray # 有效粘度
    magUb: np.ndarray # 速度大小
    gamma: np.ndarray # 阻力系数
    
    # 计算项
    L_terms: Dict[str, np.ndarray] = None
    R_terms: Dict[str, np.ndarray] = None

# ============================================================================
# 主分析类
# ============================================================================

class TurbidityCurrentAnalyzer:
    def __init__(self):
        # self.sol = '/media/amber/53EA-E81F/PhD/case231020_5'
        # self.output_dir = "/home/amber/postpro/u_vorticity/tc3d_d23_orginal"
        self.sol = '/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/3D/case230209_2'
        self.output_dir = "/home/amber/postpro/u_vorticity/tc3d_d23_0209_2"
        self.rho_f = 1000.0
        self.times = [15]
        self.z_slice = 0.135         # Z 方向切片位置
        self.alpha_threshold = 1e-4  # 用于定义头部的阈值
        self.y_min = 0.01            # 避免底边界干扰
        self.vector_calc = VectorCalculus()
        self.finite_diff = FiniteDifference()
        self.ALPHA_CONTOUR_PARAMS = {
            'levels': [1e-5],
            'colors': 'black',
            'linewidths': 2,
            'linestyles': 'dashed',
            'zorder': 3
        }
        self.X_LIM = (0.0, 4.0)
        self.Y_LIM = (0.0, 0.3)
        self.FIG_SIZE = (40, 8)
        
        # 绘图额外设置
        self.fig_height = 0.3        # 绘图 y 轴上限
        self.cmap = 'coolwarm'       # 较浅且平滑的红蓝配色，视觉更轻快屏柔和

    def _compare_gradients(self, data: TimeStepData):
        """对比自定义梯度计算与 OpenFOAM 原始数据，分析边界误差"""
        # 1. 计算我们的梯度 (3, 3, nx, ny, nz)
        # data.gradUb 现在是 (9, nx, ny, nz)，第 0 轴是 9 个张量分量。
        # - reshape(3, 3, nx, ny, nz, order='C') 是把这 9 个分量轴拆成两个轴 (3,3)。
        grad_calc = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.Ub)
        
        # 2. 这里的 data.gradUb 是 (9, nx, ny, nz)，转为 (3, 3, nx, ny, nz)
        grad_of = data.gradUb.reshape(3, 3, *data.X.shape, order='C')
        grad_of_jac = np.swapaxes(grad_of, 0, 1)                       # dU_i/dx_j
        

        # 3. 计算整体误差
        diff = grad_calc - grad_of_jac
        # diff = grad_calc-grad_of
        print(f"\n{'='*20} Gradient Verification (t={data.time}) {'='*20}")
        
        for i, comp_i in enumerate(['x', 'y', 'z']):
            for j, comp_j in enumerate(['x', 'y', 'z']):
                term_of = grad_of[i, j]
                term_diff = diff[i, j]
                
                # 计算 L2 相对误差
                ref_val = np.linalg.norm(term_of)
                l2_err = np.linalg.norm(term_diff) / (ref_val + 1e-10)
                print(f"  dU{comp_i}/d{comp_j}: Relative L2 Error = {l2_err:.4e}")
        
        # 4. 边界分析 - 重点查看 y=min (地面) 和 x=max (前端)
        print(f"\n  Boundary Error Analysis (Mean Absolute Difference):")
        # 底边界 (y 轴索引 0)
        y_min_err = np.mean(np.abs(diff[:, :, :, 0, :]))
        print(f"    - Bottom Boundary (y_min): {y_min_err:.4e}")
        # 侧边界 (z 轴索引 0 和 -1)
        z_edge_err = np.mean(np.abs(diff[:, :, :, :, [0, -1]]))
        print(f"    - Side Boundaries (z_edges): {z_edge_err:.4e}")
        # 内部点 (排除最外层点)
        inner_err = np.mean(np.abs(diff[:, :, 1:-1, 1:-1, 1:-1]))
        print(f"    - Interior Points:         {inner_err:.4e}")
        print(f"{'='*65}\n")

    def _plot_transport_terms(self, data: TimeStepData, time_v: float, head_x: float):
        """为每个 L 和 R 项生成 Z 方向切片的云图"""
        import matplotlib.pyplot as plt
        
        # 统一设置字体大小
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 22,
            'axes.labelsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20
        })
        
        plot_dir = os.path.join(self.output_dir, f"plots_t{time_v}")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Generating contour plots in {plot_dir}...")
        
        # 1. 找到最近的 Z 切面索引
        z_idx = np.argmin(np.abs(data.Z[0, 0, :] - self.z_slice))
        print(f"Selected Z slice at z={data.Z[0, 0, z_idx]:.3f} (index {z_idx}) for plotting")
        real_z = data.Z[0, 0, z_idx]
        
        # 2. 提取切面网格
        X_slice = data.X[:, :, z_idx]
        Y_slice = data.Y[:, :, z_idx]
        alpha_slice = data.alpha_A[:, :, z_idx]
        
        # 3. 准备所有待绘制的项 (包括总和项)
        L_total = np.zeros_like(data.vorticityUb)
        for k, v in data.L_terms.items():
            if k != 'L0': L_total += v
            
        R_total = np.zeros_like(data.vorticityUb)
        for v in data.R_terms.values():
            R_total += v
            
        all_plots = {**data.L_terms, **data.R_terms, 'L_sum': L_total, 'R_sum': R_total}
        
        for name, field in all_plots.items():
            # 跳过验证项 L0 的绘图 (速度梯度矩阵)
            if name == 'L0':
                continue
                
            # 提取数据：如果是向量 (3, nx, ny, nz) 则取 Z 分量 [2]观察那些“绕着 z 轴转”的涡量（即垂直于平面的旋转强度）。
            if field.ndim == 4:
                plot_val = field[2, :, :, z_idx]
                title_suffix = "(Z-component)"
            else:
                plot_val = field[:, :, z_idx]
                title_suffix = "(Scalar)"
                
            plt.figure(figsize=self.FIG_SIZE) # 稍微收紧比例
            
            # 排除极值干扰设置对比度
            vlimit = np.percentile(np.abs(plot_val), 98) 
            if vlimit == 0: vlimit = 1e-10
            
            # 使用 contourf 代替 pcolormesh，增加 levels 使其平滑
            # levels=100 会产生非常细腻且清晰的等值线填充
            levels = np.linspace(-vlimit, vlimit, 101)
            
            im = plt.contourf(X_slice, Y_slice, plot_val, 
                             levels=levels,
                             cmap=self.cmap,
                             extend='both')
            plt.contour(X_slice, Y_slice, alpha_slice, **self.ALPHA_CONTOUR_PARAMS)
            # 添加颜色条
            plt.colorbar(im, label='Value', aspect=20)
            
            plt.xlim(*self.X_LIM)
            plt.ylim(*self.Y_LIM)
            plt.title(f"{name} {title_suffix} at t={time_v}, z={real_z:.3f}")
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            
            # 提高保存清晰度到 300 DPI
            save_name = os.path.join(plot_dir, f"contour_{name}.png")
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"All plots saved for t={time_v}")

    def _locate_head_position(self, X: np.ndarray, Y: np.ndarray, alpha_A: np.ndarray) -> float:
        """从流向末端向前搜索，定位头部第一个满足阈值的 x 坐标"""
        # X 维度为 (nx, ny, nz)，获取唯一的 x 轴坐标并倒序
        x_axes = np.unique(X)[::-1]
        for x in x_axes:
            # 在该 x 截面寻找是否存在 alpha > threshold
            mask = (X == x) & (Y >= self.y_min) & (alpha_A > self.alpha_threshold)
            if np.any(mask):
                return x
        return np.max(X) # 没找到则返回最大值

    def _compute_L_terms(self, data: TimeStepData) -> Dict[str, np.ndarray]:
        """计算涡量输运方程左端项"""
        # L0: 使用 FiniteDifference 计算的速度梯度 (3, 3, nx, ny, nz) 用于验证项
        L0 = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.Ub)

        # L1: 密度梯度跟动能梯度的叉乘
        gradmagUb = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.magUb)
        gradke = gradmagUb * data.magUb[np.newaxis, ...] # grad(0.5*|U|^2) = U · grad(U)
        L1 = self.vector_calc.cross_product(data.gradbeta, gradke) * self.rho_f
        
        # L2: Lamb向量与密度梯度的叉乘
        lamb = self.vector_calc.cross_product(data.Ub, data.vorticityUb)
        L2 = self.vector_calc.cross_product(lamb, data.gradbeta) * self.rho_f
        
        # L3: 涡量拉伸项 (Tensor contraction)
        L3 = self.vector_calc.tensor_vector_contraction(data.gradUb, data.vorticityUb) * data.beta * self.rho_f
        
        # L4: ρ * β * (U_a · ∇)ω
        grad_vortUb = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.vorticityUb)
        L4 = self.vector_calc.tensor_vector_contraction(grad_vortUb, data.Ua) * data.beta * self.rho_f
        
        # L5: 压缩项 (Vorticity * Divergence)
        divUb = self.vector_calc.divergence(data.gradUb)
        L5 = data.beta * data.vorticityUb * divUb[np.newaxis, ...] * self.rho_f
        
        return {'L0': L0, 'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'L5': L5}

    def _compute_R_terms(self, data: TimeStepData) -> Dict[str, np.ndarray]:
        """计算扩散和粘性相关项"""
        # R1: 扩散项 beta * nueff * laplacian(vorticity)
        lap_vort = self.finite_diff.compute_laplacian_simple(data.X, data.Y, data.Z, data.vorticityUb)
        R1 = data.beta * lap_vort * self.rho_f * data.nuEffb

        # R2: 粘性项 (grad(nuEff) · laplacian(Ub))
        betanuEff = data.beta * data.nuEffb
        grad_betanuEff = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, betanuEff)
        lap_Ub = self.finite_diff.compute_laplacian_simple(data.X, data.Y, data.Z, data.Ub)
        R2 = self.vector_calc.cross_product(grad_betanuEff, lap_Ub) * self.rho_f
        
        # R3 curl of (grad(betanuEff) · grad(Ub))
        gradUb_tensor = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.Ub)
        # 这里的 dotGradient 应为向量 (3, nx, ny, nz)
        # 计算 (grad_betanuEff · ∇) Ub_i
        dotGradient = self.vector_calc.tensor_vector_contraction(gradUb_tensor, grad_betanuEff) * self.rho_f    
        R3 = self.finite_diff.compute_vorticity_simple(data.X, data.Y, data.Z, dotGradient)

        # R4 baroclinic effect (grad(nuEff) x grad(div(Ub)))
        divUb = self.vector_calc.divergence(data.gradUb)
        # 确保 divUb 是 (nx, ny, nz) 
        if divUb.ndim == 4 and divUb.shape[0] == 1:
            divUb = divUb[0]
        gradDiv = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, divUb)
        R4 = self.vector_calc.cross_product(grad_betanuEff, gradDiv) * self.rho_f

        # R5 curl of gradient of alpha.b as move with the flow
        g = grad_betanuEff
        grad_g = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, g)
        # grad_g 是 (3, 3, nx, ny, nz), data.Ub 是 (3, nx, ny, nz)
        grad_gUb = self.vector_calc.tensor_vector_contraction(grad_g, data.Ub) * self.rho_f
        R5 = self.finite_diff.compute_vorticity_simple(data.X, data.Y, data.Z, grad_gUb)

        # R6 vorticity difference due to drag force: beta * K * (omega_a - omega_b)
        vorticity_diff = data.vorticityUa - data.vorticityUb
        R6 = data.beta * vorticity_diff * data.gamma * self.rho_f

        # R7 velocity difference due to drag force: K * grad(beta) x (Ua - Ub)
        velocity_diff = data.Ua - data.Ub
        grad_Beta = self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.beta)
        R7 = self.vector_calc.cross_product(grad_Beta, velocity_diff) * data.gamma * self.rho_f

        return {'R1': R1, 'R2': R2, 'R3': R3, 'R4': R4, 'R5': R5, 'R6': R6, 'R7': R7}
    
    def _save_to_csv(self, data: TimeStepData, time_v: float, head_x: float):
        """将结构化计算结果打平并保存为 CSV，包含 L 和 R 的总和，且只保留 x <= head_x 且 z == z_slice 的部分"""
        out_path = os.path.join(self.output_dir, f"vorticity_transport_{time_v}.csv")
        
        # 找到最近的实际 Z 坐标以确保 mask 不为空
        z_idx = np.argmin(np.abs(data.Z[0, 0, :] - self.z_slice))
        real_z = data.Z[0, 0, z_idx]
        
        # 使用找到的实际 z 坐标进行过滤
        mask = (data.X <= head_x) & (data.Z == real_z)
        
        if not np.any(mask):
            print(f"Warning: No data found for z={real_z:.3f} and x<={head_x}")
            return
            
        # 基础坐标
        export_data = {
            'x': data.X[mask],
            'y': data.Y[mask],
            'z': data.Z[mask]
        }
        
        # 2. 计算 L 总和与 R 总和 (L0 是验证项，通常不计入方程总和)
        L_total = np.zeros_like(data.vorticityUb)
        for k, v in data.L_terms.items():
            if k != 'L0': L_total += v
            
        R_total = np.zeros_like(data.vorticityUb)
        for v in data.R_terms.values():
            R_total += v
            
        # 3. 合并所有计算项并由于 mask 展平
        all_terms = {**data.L_terms, **data.R_terms, 'L_sum': L_total, 'R_sum': R_total}
        
        for name, field in all_terms.items():
            if field.ndim == 3: # 标量 (nx, ny, nz)
                export_data[name] = field[mask]
            elif field.ndim == 4: # 向量 (3, nx, ny, nz)
                export_data[f"{name}_x"] = field[0][mask]
                export_data[f"{name}_y"] = field[1][mask]
                export_data[f"{name}_z"] = field[2][mask]
        
        # 4. 保存
        df = pd.DataFrame(export_data)
        df.to_csv(out_path, index=False)
        print(f"Results (up to x={head_x:.3f}) saved to CSV: {out_path}")

    def process_time_step(self, X_raw, Y_raw, Z_raw, time_v):
        # 1. 加载并排序重构数据
        print(f"Processing t={time_v}...")
        raw = {
            'U.a': fluidfoam.readvector(self.sol, str(time_v), "U.a"),
            'U.b': fluidfoam.readvector(self.sol, str(time_v), "U.b"),
            'alpha.a': fluidfoam.readscalar(self.sol, str(time_v), "alpha.a"),
            'alpha.b': fluidfoam.readscalar(self.sol, str(time_v), "alpha.b"),
            'vorticityUa': fluidfoam.readvector(self.sol, str(time_v), "vorticity_Ua"),
            'vorticityUb':fluidfoam.readvector(self.sol, str(time_v), "vorticity_Ub"),
            'gradUb': fluidfoam.readtensor(self.sol, str(time_v), "grad(U.b)"),
            'gradBeta': fluidfoam.readvector(self.sol, str(time_v), "grad(alpha.b)"),
            'nuEffb': fluidfoam.readscalar(self.sol, str(time_v), "nuEffb"),
            'magUb': fluidfoam.readscalar(self.sol, str(time_v), "mag(U.b)"),
            'gamma': fluidfoam.readscalar(self.sol, str(time_v), "K")
        }
        
        nx, ny, nz = len(np.unique(X_raw)), len(np.unique(Y_raw)), len(np.unique(Z_raw))
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))
        
        # 将展平数据转为 (nx, ny, nz) 格式
        def reshape_f(field):
            if field.ndim == 1: return field[sort_idx].reshape(nx, ny, nz)
            return field[:, sort_idx].reshape(field.shape[0], nx, ny, nz)

        data = TimeStepData(
            time=float(time_v),
            X=reshape_f(X_raw), Y=reshape_f(Y_raw), 
            Z=reshape_f(Z_raw),
            alpha_A=reshape_f(raw['alpha.a']), 
            beta=reshape_f(raw['alpha.b']),
            Ua=reshape_f(raw['U.a']), 
            Ub=reshape_f(raw['U.b']),
            vorticityUa=reshape_f(raw['vorticityUa']),
            vorticityUb=reshape_f(raw['vorticityUb']),
            gradUb=reshape_f(raw['gradUb']),
            gradbeta=reshape_f(raw['gradBeta']),
            nuEffb=reshape_f(raw['nuEffb']),
            magUb=reshape_f(raw['magUb']),
            gamma=reshape_f(raw['gamma'])
        )

        # 2. 计算输运项
        data.L_terms = self._compute_L_terms(data)
        data.R_terms = self._compute_R_terms(data)
        
        # 3. 定位头部并输出数据
        head_x = self._locate_head_position(data.X, data.Y, data.alpha_A)
        
        # 输出验证信息: 自定义速度梯度 vs OpenFOAM 速度梯度
        self._compare_gradients(data)
        
        # 保存 CSV
        self._save_to_csv(data, time_v, head_x)
        
        # 绘制云图
        self._plot_transport_terms(data, time_v, head_x)
        
        return data

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        X, Y, Z = fluidfoam.readmesh(self.sol)
        for t in self.times:
            self.process_time_step(X, Y, Z, t)

if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()
