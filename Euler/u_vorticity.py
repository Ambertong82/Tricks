import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


# ============================================================================
# 数学运算工具类 - 规范化向量和微分运算
# ============================================================================

class VectorCalculus:
    """封装所有向量和张量运算操作"""
    
    @staticmethod
    def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        批量交叉积计算
        
        Args:
            a: (3, N) 或 (3, nx, ny, nz) 数组
            b: (3, N) 或 (3, nx, ny, nz) 数组
            
        Returns:
            交叉积结果，维度与输入相同
        """
        if a.shape != b.shape:
            raise ValueError(f"Shapes must match: a={a.shape}, b={b.shape}")
        
        # 处理不同维度
        if a.ndim == 2 and a.shape[0] == 3:
            # (3, N) 情况
            cross_x = a[1] * b[2] - a[2] * b[1]
            cross_y = a[2] * b[0] - a[0] * b[2]
            cross_z = a[0] * b[1] - a[1] * b[0]
            return np.array([cross_x, cross_y, cross_z])
        
        elif a.ndim == 4 and a.shape[0] == 3:
            # (3, nx, ny, nz) 情况
            cross_x = a[1] * b[2] - a[2] * b[1]
            cross_y = a[2] * b[0] - a[0] * b[2]
            cross_z = a[0] * b[1] - a[1] * b[0]
            return np.stack([cross_x, cross_y, cross_z], axis=0)
        
        else:
            raise ValueError(f"Unsupported shape: {a.shape}")
        


    
    @staticmethod
    def dot_product_vector_gradient(gradient: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """
        向量与梯度张量的点积
        
        Args:
            gradient: (3, N) 梯度向量 [∂f/∂x, ∂f/∂y, ∂f/∂z]
            tensor: (9, N) 或 (3, 3, N) 张量，按列优先存储
            
        Returns:
            点积结果 (3, N)
        """
        if tensor.shape[0] == 9:
            # OpenFOAM 9分量格式转换为3x3
            tensor = tensor.reshape(3, 3, -1)
        
        drho_dx, drho_dy, drho_dz = gradient
        
        # 张量-向量点积：对列指标求和
        result_x = drho_dx * tensor[0, 0] + drho_dy * tensor[0, 1] + drho_dz * tensor[0, 2]
        result_y = drho_dx * tensor[1, 0] + drho_dy * tensor[1, 1] + drho_dz * tensor[1, 2]
        result_z = drho_dx * tensor[2, 0] + drho_dy * tensor[2, 1] + drho_dz * tensor[2, 2]
        
        return np.array([result_x, result_y, result_z])
        
    @staticmethod  
    def tensor_vector_contraction(tensor: np.ndarray, vector: np.ndarray,
                                contraction_axis: int = 1) -> np.ndarray:
        """
        张量与向量的缩并
        
        Args:
            tensor: (..., 3, ...) 张量
            vector: (3, ...) 向量
            contraction_axis: 缩并的轴（在张量中）
            
        Returns:
            缩并结果
        """
        # 处理OpenFOAM 9分量格式
        if tensor.shape[0] == 9:
            tensor = tensor.reshape(3, 3, -1)
        
        # 构建正确的 einsum 字符串
        # 对于 (3, 3, N) 张量和 (3, N) 向量：
        # 我们想要：tensor[i,j,n] * vector[j,n] -> result[i,n]
        
        if contraction_axis == 0:
            # 缩并第一个轴：tensor[i,j,n] * vector[i,n] -> result[j,n] 
            return np.einsum('ijn,in->jn', tensor, vector)
        elif contraction_axis == 1:
            # 缩并第二个轴：tensor[i,j,n] * vector[j,n] -> result[i,n] (这是物理需要的)
            return np.einsum('ijn,jn->in', tensor, vector)
        else:
            # 对于其他情况
            rank = tensor.ndim
            tensor_idx = ''.join([chr(97+i) for i in range(rank)])
            vector_idx = chr(97+contraction_axis) + tensor_idx[contraction_axis+1:]
            result_idx = tensor_idx.replace(chr(97+contraction_axis), '')
            
            return np.einsum(f'{tensor_idx},{vector_idx}->{result_idx}', 
                            tensor, vector)

        
    @staticmethod
    def divergence_velocity(grad_U: np.ndarray, grad_format: str = 'openfoam') -> np.ndarray:
        """
        计算速度散度
        
        Args:
            grad_U: 速度梯度张量
            grad_format: 'openfoam' (9分量) 或 'matrix' (3x3)
            
        Returns:
            散度场 (N,)
        """
        if grad_format == 'openfoam':
            # OpenFOAM格式: [du/dx, dv/dx, dw/dx, du/dy, dv/dy, dw/dy, du/dz, dv/dz, dw/dz]
            # 对角线元素索引: 0, 4, 8
            return grad_U[0] + grad_U[4] + grad_U[8]
        elif grad_format == 'matrix':
            # 矩阵格式 (3, 3, N)
            return grad_U[0, 0] + grad_U[1, 1] + grad_U[2, 2]
        else:
            raise ValueError(f"Unknown gradient format: {grad_format}")


# ============================================================================
# 微分算子类 - 规范化数值微分计算
# ============================================================================

class FiniteDifference:
    """结构化网格上的数值微分计算"""
    
    @staticmethod
    def compute_gradient_1d(field: np.ndarray, coords: np.ndarray, 
                           axis: int) -> np.ndarray:
        """
        一维梯度计算（中心差分）
        
        Args:
            field: 场变量
            coords: 坐标
            axis: 计算梯度的轴
            
        Returns:
            梯度场
        """
        gradient = np.zeros_like(field)
        n = field.shape[axis]
        
        # 内部点用中心差分
        slice_before = [slice(None)] * field.ndim
        slice_center = [slice(None)] * field.ndim
        slice_after = [slice(None)] * field.ndim
        
        # 中心点
        for i in range(1, n-1):
            slice_before[axis] = i-1
            slice_center[axis] = i
            slice_after[axis] = i+1
            
            coords_before = coords[tuple(slice_before)]
            coords_center = coords[tuple(slice_center)]
            coords_after = coords[tuple(slice_after)]
            
            dx_before = coords_center - coords_before
            dx_after = coords_after - coords_center
            
            gradient[tuple(slice_center)] = (
                (field[tuple(slice_after)] - field[tuple(slice_center)]) / dx_after +
                (field[tuple(slice_center)] - field[tuple(slice_before)]) / dx_before
            ) * 0.5
        
        # 边界点用单边差分
        slice_before[axis] = 0
        slice_center[axis] = 1
        slice_after[axis] = 0
        gradient[tuple(slice_before)] = (field[1] - field[0]) / (coords[1] - coords[0])
        
        slice_before[axis] = -1
        slice_center[axis] = -2
        slice_after[axis] = -1
        gradient[tuple(slice_before)] = (field[-1] - field[-2]) / (coords[-1] - coords[-2])
        
        return gradient
    
    @staticmethod
    def compute_curl_with_boundary(coords_x: np.ndarray, coords_y: np.ndarray, coords_z: np.ndarray,
                                A_field: np.ndarray) -> np.ndarray:
        """
        完整边界处理：内部用中心差分，边界用一阶差分
        """
        nx, ny, nz = coords_x.shape
        curl_field = np.zeros((3, nx, ny, nz))
        
        Ax, Ay, Az = A_field
        
        def compute_derivative_at(i, j, k, field, direction):
            """在指定点计算指定方向的导数"""
            if direction == 'x':
                # x方向导数 ∂/∂x
                if i == 0:
                    # 左边界：前向差分
                    dx = coords_x[1, j, k] - coords_x[0, j, k]
                    return (field[1, j, k] - field[0, j, k]) / dx if dx != 0 else 0
                elif i == nx-1:
                    # 右边界：后向差分
                    dx = coords_x[i, j, k] - coords_x[i-1, j, k]
                    return (field[i, j, k] - field[i-1, j, k]) / dx if dx != 0 else 0
                else:
                    # 内部点：中心差分
                    dx_total = coords_x[i+1, j, k] - coords_x[i-1, j, k]
                    return (field[i+1, j, k] - field[i-1, j, k]) / dx_total if dx_total != 0 else 0
                    
            elif direction == 'y':
                # y方向导数 ∂/∂y
                if j == 0:
                    # 下边界：前向差分
                    dy = coords_y[i, 1, k] - coords_y[i, 0, k]
                    return (field[i, 1, k] - field[i, 0, k]) / dy if dy != 0 else 0
                elif j == ny-1:
                    # 上边界：后向差分
                    dy = coords_y[i, j, k] - coords_y[i, j-1, k]
                    return (field[i, j, k] - field[i, j-1, k]) / dy if dy != 0 else 0
                else:
                    # 内部点：中心差分
                    dy_total = coords_y[i, j+1, k] - coords_y[i, j-1, k]
                    return (field[i, j+1, k] - field[i, j-1, k]) / dy_total if dy_total != 0 else 0
                    
            elif direction == 'z':
                # z方向导数 ∂/∂z
                if k == 0:
                    # 后边界：前向差分
                    dz = coords_z[i, j, 1] - coords_z[i, j, 0]
                    return (field[i, j, 1] - field[i, j, 0]) / dz if dz != 0 else 0
                elif k == nz-1:
                    # 前边界：后向差分
                    dz = coords_z[i, j, k] - coords_z[i, j, k-1]
                    return (field[i, j, k] - field[i, j, k-1]) / dz if dz != 0 else 0
                else:
                    # 内部点：中心差分
                    dz_total = coords_z[i, j, k+1] - coords_z[i, j, k-1]
                    return (field[i, j, k+1] - field[i, j, k-1]) / dz_total if dz_total != 0 else 0
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 计算所有需要的导数
                    dAz_dy = compute_derivative_at(i, j, k, Az, 'y')
                    dAy_dz = compute_derivative_at(i, j, k, Ay, 'z')
                    
                    dAx_dz = compute_derivative_at(i, j, k, Ax, 'z')
                    dAz_dx = compute_derivative_at(i, j, k, Az, 'x')
                    
                    dAy_dx = compute_derivative_at(i, j, k, Ay, 'x')
                    dAx_dy = compute_derivative_at(i, j, k, Ax, 'y')
                    
                    # 旋度
                    curl_field[0, i, j, k] = dAz_dy - dAy_dz
                    curl_field[1, i, j, k] = dAx_dz - dAz_dx
                    curl_field[2, i, j, k] = dAy_dx - dAx_dy
        
        return curl_field

    
    @staticmethod
    def _compute_scalar_laplacian_simplified(
        coords_x: np.ndarray, coords_y: np.ndarray, coords_z: np.ndarray,
        field: np.ndarray, extend: bool = False
    ) -> np.ndarray:
        """简化的标量拉普拉斯计算（统一处理边界）"""
        nx, ny, nz = coords_x.shape
        laplacian = np.zeros_like(field)
        
        # 预先计算网格间距
        # 向前差分：从i到i+1
        dx_forward = np.zeros_like(coords_x)
        dy_forward = np.zeros_like(coords_y)
        dz_forward = np.zeros_like(coords_z)
        
        dx_forward[:-1, :, :] = coords_x[1:, :, :] - coords_x[:-1, :, :]
        dy_forward[:, :-1, :] = coords_y[:, 1:, :] - coords_y[:, :-1, :]
        dz_forward[:, :, :-1] = coords_z[:, :, 1:] - coords_z[:, :, :-1]
        
        # 向后差分：从i到i-1
        dx_backward = np.zeros_like(coords_x)
        dy_backward = np.zeros_like(coords_y)
        dz_backward = np.zeros_like(coords_z)
        
        dx_backward[1:, :, :] = coords_x[1:, :, :] - coords_x[:-1, :, :]
        dy_backward[:, 1:, :] = coords_y[:, 1:, :] - coords_y[:, :-1, :]
        dz_backward[:, :, 1:] = coords_z[:, :, 1:] - coords_z[:, :, :-1]
        
        # 函数：判断是否需要使用边界差分
        def is_boundary_index(idx, dim_size, extend_flag):
            """判断是否为边界点，根据是否扩展模式"""
            if not extend_flag:
                # 不扩展：边界点用0
                return idx == 0 or idx == dim_size-1
            else:
                # 扩展模式：边界点用前向/后向差分
                return True  # 所有点都计算
        
        # 遍历所有点
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # x方向二阶导数
                    if i == 0:
                        # 左边界
                        if extend and dx_forward[0, j, k] != 0:
                            # 前向差分二阶导
                            df_dx = (field[1, j, k] - field[0, j, k]) / dx_forward[0, j, k]
                            d2f_dx2 = 2 * df_dx / (dx_forward[0, j, k])
                        else:
                            d2f_dx2 = 0
                    elif i == nx-1:
                        # 右边界
                        if extend and dx_backward[i, j, k] != 0:
                            # 后向差分二阶导
                            df_dx = (field[i, j, k] - field[i-1, j, k]) / dx_backward[i, j, k]
                            d2f_dx2 = 2 * df_dx / (dx_backward[i, j, k])
                        else:
                            d2f_dx2 = 0
                    else:
                        # 内部点
                        if dx_backward[i, j, k] != 0 and dx_forward[i, j, k] != 0:
                            df_dx_before = (field[i, j, k] - field[i-1, j, k]) / dx_backward[i, j, k]
                            df_dx_after = (field[i+1, j, k] - field[i, j, k]) / dx_forward[i, j, k]
                            dx_mid = 0.5 * (dx_backward[i, j, k] + dx_forward[i, j, k])
                            d2f_dx2 = (df_dx_after - df_dx_before) / dx_mid
                        else:
                            d2f_dx2 = 0
                    
                    # y方向二阶导数（同理）
                    if j == 0:
                        if extend and dy_forward[i, 0, k] != 0:
                            df_dy = (field[i, 1, k] - field[i, 0, k]) / dy_forward[i, 0, k]
                            d2f_dy2 = 2 * df_dy / (dy_forward[i, 0, k])
                        else:
                            d2f_dy2 = 0
                    elif j == ny-1:
                        if extend and dy_backward[i, j, k] != 0:
                            df_dy = (field[i, j, k] - field[i, j-1, k]) / dy_backward[i, j, k]
                            d2f_dy2 = 2 * df_dy / (dy_backward[i, j, k])
                        else:
                            d2f_dy2 = 0
                    else:
                        if dy_backward[i, j, k] != 0 and dy_forward[i, j, k] != 0:
                            df_dy_before = (field[i, j, k] - field[i, j-1, k]) / dy_backward[i, j, k]
                            df_dy_after = (field[i, j+1, k] - field[i, j, k]) / dy_forward[i, j, k]
                            dy_mid = 0.5 * (dy_backward[i, j, k] + dy_forward[i, j, k])
                            d2f_dy2 = (df_dy_after - df_dy_before) / dy_mid
                        else:
                            d2f_dy2 = 0
                    
                    # z方向二阶导数（同理）
                    if k == 0:
                        if extend and dz_forward[i, j, 0] != 0:
                            df_dz = (field[i, j, 1] - field[i, j, 0]) / dz_forward[i, j, 0]
                            d2f_dz2 = 2 * df_dz / (dz_forward[i, j, 0])
                        else:
                            d2f_dz2 = 0
                    elif k == nz-1:
                        if extend and dz_backward[i, j, k] != 0:
                            df_dz = (field[i, j, k] - field[i, j, k-1]) / dz_backward[i, j, k]
                            d2f_dz2 = 2 * df_dz / (dz_backward[i, j, k])
                        else:
                            d2f_dz2 = 0
                    else:
                        if dz_backward[i, j, k] != 0 and dz_forward[i, j, k] != 0:
                            df_dz_before = (field[i, j, k] - field[i, j, k-1]) / dz_backward[i, j, k]
                            df_dz_after = (field[i, j, k+1] - field[i, j, k]) / dz_forward[i, j, k]
                            dz_mid = 0.5 * (dz_backward[i, j, k] + dz_forward[i, j, k])
                            d2f_dz2 = (df_dz_after - df_dz_before) / dz_mid
                        else:
                            d2f_dz2 = 0
                    
                    laplacian[i, j, k] = d2f_dx2 + d2f_dy2 + d2f_dz2
        
    @staticmethod
    def compute_vorticity_simple(coords_x: np.ndarray, coords_y: np.ndarray, coords_z: np.ndarray, 
                                U_field: np.ndarray) -> np.ndarray:
        """
        使用简化的差分计算涡量（旋度）
        
        Args:
            coords_x, coords_y, coords_z: (nx, ny, nz) 网格坐标
            U_field: (3, nx, ny, nz) 速度场
            
        Returns:
            (3, nx, ny, nz) 涡量场
        """
        nx, ny, nz = coords_x.shape
        vorticity = np.zeros((3, nx, ny, nz))
        
        Ux, Uy, Uz = U_field
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # ∂Uz/∂y - ∂Uy/∂z
                    if j == 0:
                        dUz_dy = (Uz[i, 1, k] - Uz[i, 0, k]) / (coords_y[i, 1, k] - coords_y[i, 0, k])
                    elif j == ny-1:
                        dUz_dy = (Uz[i, j, k] - Uz[i, j-1, k]) / (coords_y[i, j, k] - coords_y[i, j-1, k])
                    else:
                        dUz_dy = (Uz[i, j+1, k] - Uz[i, j-1, k]) / (coords_y[i, j+1, k] - coords_y[i, j-1, k])
                    
                    if k == 0:
                        dUy_dz = (Uy[i, j, 1] - Uy[i, j, 0]) / (coords_z[i, j, 1] - coords_z[i, j, 0])
                    elif k == nz-1:
                        dUy_dz = (Uy[i, j, k] - Uy[i, j, k-1]) / (coords_z[i, j, k] - coords_z[i, j, k-1])
                    else:
                        dUy_dz = (Uy[i, j, k+1] - Uy[i, j, k-1]) / (coords_z[i, j, k+1] - coords_z[i, j, k-1])
                    
                    vorticity[0, i, j, k] = dUz_dy - dUy_dz
                    
                    # ∂Ux/∂z - ∂Uz/∂x
                    if k == 0:
                        dUx_dz = (Ux[i, j, 1] - Ux[i, j, 0]) / (coords_z[i, j, 1] - coords_z[i, j, 0])
                    elif k == nz-1:
                        dUx_dz = (Ux[i, j, k] - Ux[i, j, k-1]) / (coords_z[i, j, k] - coords_z[i, j, k-1])
                    else:
                        dUx_dz = (Ux[i, j, k+1] - Ux[i, j, k-1]) / (coords_z[i, j, k+1] - coords_z[i, j, k-1])
                    
                    if i == 0:
                        dUz_dx = (Uz[1, j, k] - Uz[0, j, k]) / (coords_x[1, j, k] - coords_x[0, j, k])
                    elif i == nx-1:
                        dUz_dx = (Uz[i, j, k] - Uz[i-1, j, k]) / (coords_x[i, j, k] - coords_x[i-1, j, k])
                    else:
                        dUz_dx = (Uz[i+1, j, k] - Uz[i-1, j, k]) / (coords_x[i+1, j, k] - coords_x[i-1, j, k])
                    
                    vorticity[1, i, j, k] = dUx_dz - dUz_dx
                    
                    # ∂Uy/∂x - ∂Ux/∂y
                    if i == 0:
                        dUy_dx = (Uy[1, j, k] - Uy[0, j, k]) / (coords_x[1, j, k] - coords_x[0, j, k])
                    elif i == nx-1:
                        dUy_dx = (Uy[i, j, k] - Uy[i-1, j, k]) / (coords_x[i, j, k] - coords_x[i-1, j, k])
                    else:
                        dUy_dx = (Uy[i+1, j, k] - Uy[i-1, j, k]) / (coords_x[i+1, j, k] - coords_x[i-1, j, k])
                    
                    if j == 0:
                        dUx_dy = (Ux[i, 1, k] - Ux[i, 0, k]) / (coords_y[i, 1, k] - coords_y[i, 0, k])
                    elif j == ny-1:
                        dUx_dy = (Ux[i, j, k] - Ux[i, j-1, k]) / (coords_y[i, j, k] - coords_y[i, j-1, k])
                    else:
                        dUx_dy = (Ux[i, j+1, k] - Ux[i, j-1, k]) / (coords_y[i, j+1, k] - coords_y[i, j-1, k])
                    
                    vorticity[2, i, j, k] = dUy_dx - dUx_dy
        
        return vorticity
    
    @staticmethod
    def compute_gradient_simple(coords_x: np.ndarray, coords_y: np.ndarray, coords_z: np.ndarray, 
                               field: np.ndarray) -> np.ndarray:
        """
        使用简化的差分计算梯度
        
        Args:
            coords_x, coords_y, coords_z: (nx, ny, nz) 网格坐标
            field: (nx, ny, nz) 标量场
            
        Returns:
            (3, nx, ny, nz) 梯度场 [∂f/∂x, ∂f/∂y, ∂f/∂z]
        """
        nx, ny, nz = coords_x.shape
        gradient = np.zeros((3, nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # ∂f/∂x
                    if i == 0:
                        df_dx = (field[1, j, k] - field[0, j, k]) / (coords_x[1, j, k] - coords_x[0, j, k])
                    elif i == nx-1:
                        df_dx = (field[i, j, k] - field[i-1, j, k]) / (coords_x[i, j, k] - coords_x[i-1, j, k])
                    else:
                        df_dx = (field[i+1, j, k] - field[i-1, j, k]) / (coords_x[i+1, j, k] - coords_x[i-1, j, k])
                    
                    # ∂f/∂y
                    if j == 0:
                        df_dy = (field[i, 1, k] - field[i, 0, k]) / (coords_y[i, 1, k] - coords_y[i, 0, k])
                    elif j == ny-1:
                        df_dy = (field[i, j, k] - field[i, j-1, k]) / (coords_y[i, j, k] - coords_y[i, j-1, k])
                    else:
                        df_dy = (field[i, j+1, k] - field[i, j-1, k]) / (coords_y[i, j+1, k] - coords_y[i, j-1, k])
                    
                    # ∂f/∂z
                    if k == 0:
                        df_dz = (field[i, j, 1] - field[i, j, 0]) / (coords_z[i, j, 1] - coords_z[i, j, 0])
                    elif k == nz-1:
                        df_dz = (field[i, j, k] - field[i, j, k-1]) / (coords_z[i, j, k] - coords_z[i, j, k-1])
                    else:
                        df_dz = (field[i, j, k+1] - field[i, j, k-1]) / (coords_z[i, j, k+1] - coords_z[i, j, k-1])
                    
                    gradient[0, i, j, k] = df_dx
                    gradient[1, i, j, k] = df_dy
                    gradient[2, i, j, k] = df_dz
        
        return gradient


@dataclass
class TimeStepData:
    """单个时间步的数据容器"""
    time: float
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    alpha_A: np.ndarray
    Ua_A: np.ndarray
    Ua_B: np.ndarray
    beta: np.ndarray
    gradU: np.ndarray
    gradUb: np.ndarray
    vorticity: np.ndarray
    gradbeta: np.ndarray
    gradvorticity: np.ndarray
    magUb: np.ndarray
    gradmagUb: np.ndarray
    nutb: np.ndarray
    gradBetaNueff: np.ndarray
    
    # 计算得到的结果
    L_terms: Dict[str, np.ndarray] = None
    R_terms: Dict[str, np.ndarray] = None
    
    def __post_init__(self):
        if self.L_terms is None:
            self.L_terms = {}
        if self.R_terms is None:
            self.R_terms = {}


# ============================================================================
# 主分析类 - 规范化流体分析
# ============================================================================

class TurbidityCurrentAnalyzer:
    """浑浊流体分析器 - 规范化版本"""
    
    def __init__(self):
        # 配置参数
        self.sol = '/media/amber/53EA-E81F/PhD/case231020_5'
        self.output_dir = "/home/amber/postpro/u_vorticity/tc3d_d23_original"
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.times = [5]
        
        # 物理参数
        self.rho_f = 1000.0  # 流体密度 kg/m³
        self.nu_molecular = 1e-6  # 分子运动粘度
        
        # 可视化参数
        self.FIG_SIZE = (40, 8)
        self.X_LIM = (0.0, 1.6)
        self.Y_LIM = (0.0, 0.3)
        self.Height = 0.3
        self.colorset = 'fuchsia'
        
        self._setup_plotting()
        
        # 数学工具实例
        self.vector_calc = VectorCalculus()
        self.finite_diff = FiniteDifference()
    
    def _setup_plotting(self):
        """设置绘图参数"""
        plt.rcParams.update({
            'font.size': 36,
            'axes.titlesize': 36,
            'axes.labelsize': 32,
            'xtick.labelsize': 32,
            'ytick.labelsize': 32,
            'legend.fontsize': 32
        })
        
        self.alpha_contour_params = {
            'levels': [1e-5],
            'colors': 'black',
            'linewidths': 2,
            'linestyles': 'dashed',
            'zorder': 3
        }
        
        self.alpha_contour_params2 = {
            'levels': [1e-3],
            'colors': 'blueviolet',
            'linewidths': 2,
            'linestyles': 'dashed',
            'zorder': 3
        }
    
    def _load_fluidfoam_data(self, time_v: float) -> Dict[str, np.ndarray]:
        """
        从OpenFOAM加载数据
        
        Args:
            time_v: 时间步
            
        Returns:
            数据字典
        """
        print(f"Loading data for time = {time_v}")
        
        try:
            data = {}
            data['U_a'] = fluidfoam.readvector(self.sol, str(time_v), "U.a")
            data['U_b'] = fluidfoam.readvector(self.sol, str(time_v), "U.b")
            data['alpha'] = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
            data['beta'] = fluidfoam.readscalar(self.sol, str(time_v), "alpha.b")
            data['gradUa'] = fluidfoam.readtensor(self.sol, str(time_v), "grad(U.a)")
            data['vorticity'] = fluidfoam.readvector(self.sol, str(time_v), "vorticity")
            data['gradbeta'] = fluidfoam.readvector(self.sol, str(time_v), "grad(alpha.b)")
            data['gradvorticity'] = fluidfoam.readtensor(self.sol, str(time_v), "grad(vorticity)")
            data['gradUb'] = fluidfoam.readtensor(self.sol, str(time_v), "grad(U.b)")
            data['magUb'] = fluidfoam.readscalar(self.sol, str(time_v), "mag(U.b)")
            data['gradmagUb'] = fluidfoam.readvector(self.sol, str(time_v), "grad(mag(U.b))")
            data['nutb'] = fluidfoam.readscalar(self.sol, str(time_v), "nut.b")
            data['gradNueff'] = fluidfoam.readvector(self.sol, str(time_v), "grad(nueff.b)")
            data['nueffb'] = fluidfoam.readscalar(self.sol, str(time_v), "nuEffb")
            
            return data
            
        except Exception as e:
            print(f"Error loading data for time {time_v}: {e}")
            raise
    
    def _create_time_step_data(self, time_v: float, X: np.ndarray, Y: np.ndarray, 
                              Z: np.ndarray, data_dict: Dict) -> TimeStepData:
        """
        创建时间步数据对象
        
        Args:
            time_v: 时间
            X, Y, Z: 网格坐标
            data_dict: 数据字典
            
        Returns:
            TimeStepData对象
        """
        return TimeStepData(
            time=time_v,
            X=X, Y=Y, Z=Z,
            alpha_A=data_dict['alpha'],
            Ua_A=data_dict['U_a'],
            Ua_B=data_dict['U_b'],
            beta=data_dict['beta'],
            gradUa=data_dict['gradUa'],
            gradUb=data_dict['gradUb'],
            vorticity=data_dict['vorticity'],
            gradbeta=data_dict['gradbeta'],
            gradvorticity=data_dict['gradvorticity'],
            magUb=data_dict['magUb'],
            gradmagUb=data_dict['gradmagUb'],
            nutb=data_dict['nutb'],
            gradNueff=data_dict['gradNueff'],
            nuEffb = data_dict['nueffb']
        )
    

    
    def _compute_L_terms(self, data: TimeStepData) -> Dict[str, np.ndarray]:
        """
        计算L项（涡量输运方程的左端项）
        
        Args:
            data: 时间步数据
            
        Returns:
            L项字典
        """

        
        # L1: 密度梯度与动能梯度的叉乘
        kinetic_grad = data.magUb * data.gradmagUb
        # L1 = self.vector_calc.cross_product(data.gradbeta, kinetic_grad) * self.rho_f
        L1 = self.vector_calc.vorticity_calculate(data.Ua_B)
        
        # L2: Lamb向量与密度梯度的叉乘
        lamb_vector = self.vector_calc.cross_product(data.Ua_B, data.vorticity)
        L2 = self.vector_calc.cross_product(lamb_vector, data.gradbeta) * self.rho_f
        
        # L3: 涡量拉伸项
        stretching = self.vector_calc.tensor_vector_contraction(data.gradUb, data.vorticity)
        L3 = stretching * data.beta * self.rho_f
        
        # L4: 涡量对流项
        convection = self.vector_calc.tensor_vector_contraction(data.gradvorticity, data.Ua_A)
        L4 = convection * data.beta * self.rho_f
        
        # L5: 压缩项
        divergence = self.vector_calc.divergence_velocity(data.gradUb, grad_format='openfoam')
        L5 = data.beta * data.vorticity * divergence[np.newaxis, :] * self.rho_f
        
        return {
            'L1': L1,
            'L2': L2,
            'L3': L3,
            'L4': L4,
            'L5': L5
        }
    
    def _compute_R_terms(self, data: TimeStepData, nueff: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算R项（涡量输运方程的右端项）
        
        Args:
            data: 时间步数据
            nueff: 有效粘度
            
        Returns:
            R项字典
        """
        # R1: 扩散项
        laplacian_vorticity = self.finite_diff.compute_laplacian(
            data.X, data.Y, data.Z, data.vorticity, is_vector=True
        )
        R1 = data.beta * laplacian_vorticity * self.rho_f * nueff
        
        # R2: 粘度梯度与速度拉普拉斯的叉乘
        laplacian_vel = self.finite_diff.compute_laplacian(
            data.X, data.Y, data.Z, data.Ua_B, is_vector=True
        )
        R2 = self.vector_calc.cross_product(data.gradBetaNueff, laplacian_vel) * self.rho_f
        
        # R3: 粘度梯度与速度梯度点积的旋度
        gradBetaNueff_dot_gradVel = self.vector_calc.dot_product_vector_gradient(
            data.gradBetaNueff, data.gradUb
        )
        R3 = self.finite_diff.compute_curl(
            data.X, data.Y, data.Z, gradBetaNueff_dot_gradVel
        ) * self.rho_f
        
        # R4: 交叉项
        divergence = self.vector_calc.divergence_velocity(data.gradUb, grad_format='openfoam')
        R4 = self.vector_calc.cross_product(data.gradBetaNueff, laplacian_vel) * divergence[np.newaxis, :] * self.rho_f
        
        return {
            'R1': R1,
            'R2': R2,
            'R3': R3,
            'R4': R4
        }
    
    def _locate_head_position(self, X: np.ndarray, Y: np.ndarray, alpha_A: np.ndarray) -> float:
        """
        定位头部位置
        
        Args:
            X, Y: 坐标
            alpha_A: 体积分数
            
        Returns:
            头部x坐标
        """
        for x in np.unique(X):
            mask = (X == x) & (Y >= self.y_min) & (alpha_A > self.alpha_threshold)
            if np.any(mask):
                return x
        
        print("Warning: No head found")
        return None
    
    def _extract_slice_at_z(self, data: TimeStepData, z_slice: float = 0.135) -> TimeStepData:
        """
        提取Z方向的切片
        
        Args:
            data: 原始数据
            z_slice: Z切片位置
            
        Returns:
            切片数据
        """
        select_mask = (data.Z == z_slice)
        
        if not np.any(select_mask):
            print(f"Warning: No points at Z = {z_slice}")
            return data
        
        # 为节省内存，只返回必要的切片数据
        sliced_data = TimeStepData(
            time=data.time,
            X=data.X[select_mask],
            Y=data.Y[select_mask],
            Z=data.Z[select_mask],
            alpha_A=data.alpha_A[select_mask],
            Ua_A=data.Ua_A[:, select_mask],
            Ua_B=data.Ua_B[:, select_mask],
            beta=data.beta[select_mask],
            gradU=data.gradU[:, select_mask],
            gradUb=data.gradUb[:, select_mask],
            vorticity=data.vorticity[:, select_mask],
            gradbeta=data.gradbeta[:, select_mask],
            gradvorticity=data.gradvorticity[:, select_mask],
            magUb=data.magUb[select_mask],
            gradmagUb=data.gradmagUb[:, select_mask],
            nutb=data.nutb[select_mask],
            gradBetaNueff=data.gradBetaNueff[:, select_mask]
        )
        
        return sliced_data
    
    def process_time_step(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                         time_v: float) -> Optional[TimeStepData]:
        """
        处理单个时间步
        
        Args:
            X, Y, Z: 网格坐标
            time_v: 时间步
            
        Returns:
            处理后的数据对象，失败时返回None
        """
        print(f"Processing time step: {time_v}")
        
        try:
            # 1. 加载数据
            raw_data = self._load_fluidfoam_data(time_v)
            
            # 2. 创建数据对象
            data = self._create_time_step_data(time_v, X, Y, Z, raw_data)
            
            # 3. 计算有效粘度
            nueff = self._compute_effective_viscosity(data.nutb)
            
            # 4. 计算L项
            L_terms = self._compute_L_terms(data)
            data.L_terms = L_terms
            
            # 5. 计算R项
            R_terms = self._compute_R_terms(data, nueff)
            data.R_terms = R_terms
            
            # 6. 定位头部位置
            head_x = self._locate_head_position(data.X, data.Y, data.alpha_A)
            if head_x is None:
                return None
            
            # 7. 提取Z切片
            sliced_data = self._extract_slice_at_z(data)
            
            # 8. 可选：保存或可视化结果
            self._save_results(sliced_data, time_v)
            
            return sliced_data
            
        except Exception as e:
            print(f"Error processing time step {time_v}: {e}")
            return None
    
    def _save_results(self, data: TimeStepData, time_v: float):
        """
        保存结果（可以根据需要实现）
        
        Args:
            data: 处理后的数据
            time_v: 时间步
        """
        # 这里可以添加保存逻辑
        output_file = os.path.join(self.output_dir, f"results_time_{time_v}.npz")
        np.savez_compressed(output_file, 
                          time=data.time,
                          X=data.X,
                          Y=data.Y,
                          Z=data.Z,
                          **data.L_terms,
                          **data.R_terms)
        print(f"Saved results to {output_file}")
    
    def run_analysis(self):
        """运行完整分析流程"""
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 加载网格
            print("Loading mesh...")
            X, Y, Z = fluidfoam.readmesh(self.sol)
            
            # 处理每个时间步
            results = {}
            for time_v in self.times:
                result = self.process_time_step(X, Y, Z, time_v)
                if result is not None:
                    results[time_v] = result
            
            print(f"Analysis complete. Results saved to: {self.output_dir}")
            print(f"Processed {len(results)} time steps successfully.")
            
            return results
            
        except Exception as e:
            print(f"Error in run_analysis: {e}")
            raise


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 初始化分析器
    analyzer = TurbidityCurrentAnalyzer()
    
    # 运行分析
    results = analyzer.run_analysis()
    
    # 可选：生成汇总报告
    if results:
        print("\nAnalysis Summary:")
        for time_v, data in results.items():
            print(f"  Time {time_v}: X range [{data.X.min():.3f}, {data.X.max():.3f}], "
                  f"Points: {len(data.X)}")
