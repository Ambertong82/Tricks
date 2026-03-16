import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from vorticity_transport.diagnostics import compare_gradients
from vorticity_transport.plotting import plot_transport_terms
from vorticity_transport.compute_l_terms import compute_l_terms
from vorticity_transport.compute_r_terms import compute_r_terms
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
    def tensor_vector_contraction(tensor: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """二阶张量与向量缩并：result_i = tensor_ij * vector_j。"""
        if tensor.ndim < 2 or tensor.shape[0] != 3 or tensor.shape[1] != 3:
            raise ValueError(
                f"tensor_vector_contraction 要求 tensor 形状为 (3,3,...)，当前为 {tensor.shape}"
            )
        if vector.ndim < 1 or vector.shape[0] != 3:
            raise ValueError(
                f"tensor_vector_contraction 要求 vector 形状为 (3,...)，当前为 {vector.shape}"
            )
        return np.einsum('ij...,j...->i...', tensor, vector)
        
    @staticmethod
    def divergence(grad_U: np.ndarray) -> np.ndarray:
        """计算散度 trace(grad_U)"""
        if grad_U.shape[0] == 9:
            return grad_U[0] + grad_U[4] + grad_U[8]
        return grad_U[0, 0] + grad_U[1, 1] + grad_U[2, 2]

# ============================================================================
# 微分算子类 - 规范化数值微分计算 (支持非均匀网格)
# 关于出现的所有速度梯度，全部采用本程序计算的数值梯度，避免直接使用 OpenFOAM 输出的梯度数据，确保一致性和可控性。
# 即输出的gradU[i,j]=dui/dxj，符合雅可比矩阵的约定（行索引 i 是速度分量，列索引 j 是空间方向）。这样在后续计算中，无论是计算涡量还是 Lamb 向量，都可以直接使用这个统一格式的梯度数据，避免了混淆和错误。
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
    def compute_second_derivative_scalar(cx,cy,cz,f):
        """计算标量场 Hessian，返回形状 (3, 3, nx, ny, nz)。"""
        if f.ndim != 3:
            raise ValueError(f"compute_second_derivative_scalar 仅支持标量场 (nx,ny,nz)，当前为 {f.shape}")

        ax, ay, az = FiniteDifference._get_axes(cx, cy, cz)
        d2f_dx2 = np.gradient(np.gradient(f, ax, axis=0), ax, axis=0)
        d2f_dy2 = np.gradient(np.gradient(f, ay, axis=1), ay, axis=1)
        d2f_dz2 = np.gradient(np.gradient(f, az, axis=2), az, axis=2)

        df_dx = np.gradient(f, ax, axis=0)
        df_dy = np.gradient(f, ay, axis=1)
        df_dz = np.gradient(f, az, axis=2)

        d2f_dxdy = np.gradient(df_dx, ay, axis=1)
        d2f_dxdz = np.gradient(df_dx, az, axis=2)
        d2f_dydz = np.gradient(df_dy, az, axis=2)

        d2f_dydx = np.gradient(df_dy, ax, axis=0)
        d2f_dzdx = np.gradient(df_dz, ax, axis=0)
        d2f_dzdy = np.gradient(df_dz, ay, axis=1)

        d2f_xy = 0.5 * (d2f_dxdy + d2f_dydx)
        d2f_xz = 0.5 * (d2f_dxdz + d2f_dzdx)
        d2f_yz = 0.5 * (d2f_dydz + d2f_dzdy)

        return np.stack([
            np.stack([d2f_dx2, d2f_xy,  d2f_xz], axis=0),
            np.stack([d2f_xy,  d2f_dy2, d2f_yz], axis=0),
            np.stack([d2f_xz,  d2f_yz,  d2f_dz2], axis=0),
        ], axis=0)



    @staticmethod
    def compute_second_derivative_simple(cx, cy, cz, f):
        """计算各个方向的纯二阶导 (∂²f/∂x², ∂²f/∂y², ∂²f/∂z²)"""
        # 这是为了laplacian项准备的二阶导数计算函数，返回一个包含三个分量的数组，
        # 每个分量对应一个空间方向的二阶导数。
        # 并不是Hessian矩阵
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
        self.sol = '/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/NEW/case230311_1'
        self.output_dir = "/home/amber/postpro/u_vorticity/tc3d_d23_0311_1"
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

        self.enabale_gradient_comparison = False # 是否启用梯度对比分析

    
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

    def _compute_gradUb(self, data: TimeStepData) -> np.ndarray:
        """统一计算速度梯度，供 L/R 项复用。"""
        return self.finite_diff.compute_gradient_simple(data.X, data.Y, data.Z, data.Ub)

    def _compute_L_terms(self, data: TimeStepData, gradUb: np.ndarray) -> Dict[str, np.ndarray]:
        """计算涡量输运方程左端项。"""
        return compute_l_terms(data, gradUb, self.finite_diff, self.vector_calc, self.rho_f)

    def _compute_R_terms(self, data: TimeStepData, gradUb: np.ndarray) -> Dict[str, np.ndarray]:
        """计算扩散和粘性相关项。"""
        return compute_r_terms(data, gradUb, self.finite_diff, self.vector_calc, self.rho_f)
    
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
            'gradBeta': fluidfoam.readvector(self.sol, str(time_v), "grad(alpha.b)"),
            'nuEffb': fluidfoam.readscalar(self.sol, str(time_v), "nuEffb"),
            'magUb': fluidfoam.readscalar(self.sol, str(time_v), "mag(U.b)"),
            'gamma': fluidfoam.readscalar(self.sol, str(time_v), "K")
        }

        gradUb_raw = None
        try:
            gradUb_raw = fluidfoam.readtensor(self.sol, str(time_v), "grad(U.b)")
            print("grad(U.b) found: using OpenFOAM gradient data for diagnostics.")
        except Exception:
            print("grad(U.b) not found: fallback to numerical gradient, continuing analysis.")
        
        nx, ny, nz = len(np.unique(X_raw)), len(np.unique(Y_raw)), len(np.unique(Z_raw))
        sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))
        
        # 将展平数据转为 (nx, ny, nz) 格式
        def reshape_f(field):
            if field.ndim == 1: return field[sort_idx].reshape(nx, ny, nz)
            return field[:, sort_idx].reshape(field.shape[0], nx, ny, nz)

        X = reshape_f(X_raw)
        Y = reshape_f(Y_raw)
        Z = reshape_f(Z_raw)
        Ua = reshape_f(raw['U.a'])
        Ub = reshape_f(raw['U.b'])

        # 如果不存在 OpenFOAM 输出的 grad(U.b)，则回退到本程序数值梯度。
        if gradUb_raw is not None:
            gradUb_data = reshape_f(gradUb_raw)
        else:
            gradUb_num = self.finite_diff.compute_gradient_simple(X, Y, Z, Ub)  # (3,3,nx,ny,nz)
            gradUb_data = np.swapaxes(gradUb_num, 0, 1).reshape(9, *X.shape, order='C')

        data = TimeStepData(
            time=float(time_v),
            X=X, Y=Y, 
            Z=Z,
            alpha_A=reshape_f(raw['alpha.a']), 
            beta=reshape_f(raw['alpha.b']),
            Ua=Ua, 
            Ub=Ub,
            vorticityUa=reshape_f(raw['vorticityUa']),
            vorticityUb=reshape_f(raw['vorticityUb']),
            gradUb=gradUb_data,
            gradbeta=reshape_f(raw['gradBeta']),
            nuEffb=reshape_f(raw['nuEffb']),
            magUb=reshape_f(raw['magUb']),
            gamma=reshape_f(raw['gamma'])
        )

        # 2. 计算输运项
        gradUb = self._compute_gradUb(data)
        data.L_terms = self._compute_L_terms(data, gradUb)
        data.R_terms = self._compute_R_terms(data, gradUb)
        
        # 3. 定位头部并输出数据
        head_x = self._locate_head_position(data.X, data.Y, data.alpha_A)
        
        # 输出验证信息: 自定义速度梯度 vs OpenFOAM 速度梯度
        if self.enabale_gradient_comparison:
            compare_gradients(data, self.finite_diff)
        
        # 保存 CSV
        self._save_to_csv(data, time_v, head_x)
        
        # 绘制云图
        plot_transport_terms(self, data, time_v, head_x)
        
        return data

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        X, Y, Z = fluidfoam.readmesh(self.sol)
        for t in self.times:
            self.process_time_step(X, Y, Z, t)

if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()
