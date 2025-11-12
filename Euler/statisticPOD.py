import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from scipy.ndimage import label
from scipy.ndimage import label, binary_closing
from matplotlib.patches import Ellipse


class TurbidityCurrentAnalyzer:
    def __init__(self):
        # === 新增时间戳 ===
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # === 用时间戳定义输出文件名 ===
        self.vortex_output_file = f"vortex_properties_{self.timestamp}.csv"
        self.plot_prefix = f"plot_{self.timestamp}_"  # 用于图像文件名
        # self.output_prefix = output_prefix if output_prefix else "default_prefix"
        # Configuration parameters
        # self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
        # self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090912_1"
        # self.sol = '/media/amber/53EA-E81F/PhD/case231020_5'
        self.sol = '/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case091020_5'

        self.output_dir = "/home/amber/postpro/u_umean_fine_tc3dmiddle"
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.times = [5,7,9,11]
        self.FIG_SIZE = (40, 6)
        self.X_LIM = (0.0, 2.0)
        self.Y_LIM = (0.0, 0.3)
        self.Height = 0.3
        self.colorset = 'fuchsia'
        self.Q_threshold = 0.5  # Q-criterion threshold for vortex detection
        self.lambda2_threshold = -0.1  # Lambda2 criterion threshold for vortex detection
        
        # Visualization parameters
        self.ALPHA_CONTOUR_PARAMS = {
            'levels': [1e-5],
            'colors': 'black',
            'linewidths': 2,
            'linestyles': 'dashed',
            'zorder': 3
        }
        self.ALPHA_CONTOUR_PARAMS2 = {
            'levels': [1e-3],
            'colors': 'blueviolet',
            'linewidths': 2,
            'linestyles': 'dashed',
            'zorder': 3
        }
        
        # Initialize global plot settings
        plt.rcParams.update({
            'font.size': 28,
            'axes.titlesize': 28,
            'axes.labelsize': 24,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 24
        })

    def calculate_q_criterion(self, dUx, dUy, dVx, dVy):
        """Calculate Q-criterion value"""
        S_xx = dUx
        S_xy = 0.5*(dUy + dVx)
        S_yx = S_xy
        S_yy = dVy
        
        Omega_xy = 0.5*(dVx - dUy)
        Omega_yx = -Omega_xy
        
        S_norm = S_xx**2 + S_xy**2 + S_yx**2 + S_yy**2
        Omega_norm = Omega_xy**2 + Omega_yx**2
        Q = 0.5 * (Omega_norm - S_norm)
        return np.nan_to_num(Q, nan=0, posinf=0, neginf=0)
        
    def plot_Q_contour(self, xi, yi, vorticity, ux,uy, alpha_i, time_v, 
                            positions, y_text, title, filename, levels=None, 
                            cmap='bwr'):
        # ...（其他初始化代码）
        
        # === 1. 计算 Q ===
        dy = yi[1, 0] - yi[0, 0]
        dx = xi[0, 1] - xi[0, 0]
        dUx = np.gradient(ux, dx, axis=1)
        dUy = np.gradient(ux, dy, axis=0)
        dVx = np.gradient(uy, dx, axis=1)
        dVy = np.gradient(uy, dy, axis=0)
        Q = self.calculate_q_criterion(dUx, dUy, dVx, dVy)
        
        # === 2. 检测涡旋 ===
        Q_mask = (Q > self.Q_threshold) & (alpha_i > self.alpha_threshold)
        labeled, n_vortices = label(Q_mask)
    

        # === 3. 绘制云图 ===
        plt.figure(figsize=self.FIG_SIZE)
        # 1. 绘制 Q 云图（高对比度）
        cbar=plt.contourf(xi, yi, Q, 
                    levels=np.linspace((1), (Q.max()), 128),
                    cmap='Spectral_r',
                    alpha=0.7)
        cbar=plt.colorbar(orientation='horizontal', pad=0.2)
        cbar.set_label('Q-criterion [1/s$^2$]',fontsize=12)
        cbar.set_ticks([(1), (Q.max()/2), (Q.max())])

        # 2. 绘制 alpha_i 的临界线
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        

        # 3. 标记 Q_mask 区域边界
        plt.contour(xi, yi, Q_mask, 
                    levels=[0.5], 
                    colors='white', linewidths=1)

        
        
        # === 4. 标注涡旋 ===
        vortex = []
        for vortex in vorticity:

            plt.scatter(*vortex['center'], c='r', s=50, marker='x', zorder=5)
            # Use vortex['id'] instead of the loop index 'i'
            plt.text(vortex['center'][0], vortex['center'][1], str(vortex['id']),
                    color='k', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6)    
        # print("***Detected {} vortices using Q-criterion.".format(n_vortices))
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(*self.X_LIM)
        plt.ylim(*self.Y_LIM)
        plt.title(title)
        
        # 6. 保存图像
        plt.savefig(os.path.join(self.output_dir, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()     
        plt.close()
    
    def integrate_quantities(self, ya, ua_x, alpha_vals):
        """Perform vertical integration of quantities"""
        sign_changes = np.where(np.diff(np.sign(ua_x)))[0]
        # 找出第一个（最低处）满足变化点
        for idx in sign_changes:
            if ya[idx] > 0.001 and ua_x[idx] > 0 and ua_x[idx + 1] < 0:
                max_ya_crossing_index = idx + 1  # （可选 +1，取决于是否需要变化后的位置）
                break
        else:
            max_ya_crossing_index = len(ya) - 1  # 没找到则默认取最高处
        # method 2: 取最大值位置    
        # max_ya_crossing_index = sign_changes[np.argmax(ya[sign_changes])] + 1 if len(sign_changes) > 0 else len(ya) - 1

        
        # Vectorized integration
        ua_alpha = ua_x * alpha_vals
        integral = np.trapz(ua_alpha[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU = np.trapz(ua_x[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU2 = np.trapz(ua_x[:max_ya_crossing_index]**2, ya[:max_ya_crossing_index])
        integral2 = np.trapz((ua_x[:max_ya_crossing_index] * alpha_vals[:max_ya_crossing_index])**2, ya[:max_ya_crossing_index])
        
        U = integralU2 / integralU if integralU != 0 else 0
        H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0
        H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0
        
        return U, H, ALPHA, H_depth, ya[max_ya_crossing_index]

    def calculate_perturbation_fields(self, X, Y, Ua_A, x_coords, x_U_mapping, x_U_mapping_alpha, gradbeta_x, omega_z, gradvorticity_x, beta):
        """Calculate perturbation velocity and advection fields"""
        U_perturb = Ua_A[0].copy()
        U_perturb_alpha = Ua_A[0].copy()
        U_mean_grid = np.zeros_like(omega_z)
        Umean_densitygradient = np.zeros_like(omega_z)
        Uper_densitygradient = np.zeros_like(omega_z)
        Umean_advection = np.zeros_like(omega_z)
        Uper_advection = np.zeros_like(omega_z)
        Uori_advection = Ua_A[0] * gradvorticity_x * beta

        for x in x_coords:
            if x in x_U_mapping:
                mask = (X == x)
                U_mean = x_U_mapping[x]
                U_mean_alpha = x_U_mapping_alpha[x]
                
                U_perturb[mask] = Ua_A[0][mask] - U_mean
                U_perturb_alpha[mask] = Ua_A[0][mask] - U_mean_alpha
                U_mean_grid[mask] = U_mean
                Umean_densitygradient[mask] = U_mean * gradbeta_x[mask] * 2 * omega_z[mask]
                Uper_densitygradient[mask] = U_perturb[mask] * gradbeta_x[mask] * 2 * omega_z[mask]
                Umean_advection[mask] = U_mean * gradvorticity_x[mask]*beta[mask]
                Uper_advection[mask] = U_perturb[mask] * gradvorticity_x[mask]*beta[mask]

        return U_perturb, U_perturb_alpha, Umean_densitygradient, Uper_densitygradient, Umean_advection, Uper_advection, Uori_advection

    def interpolate_fields(self, X, Y, xi, yi, fields):
        """Interpolate fields to regular grid"""
        return {name: griddata((X, Y), field, (xi, yi), method='linear') 
                for name, field in fields.items()}

    def plot_streamlines(self, xi, yi, ux, uy, color_field, alpha_i, time_v, positions, y_text, 
                        title, filename, color_label, vmin=None, vmax=None):
        """Generic streamline plotting function"""
        plt.figure(figsize=self.FIG_SIZE)
        mask = alpha_i > 1e-5  # 找出 alpha_i > 1e-5 的区域
        ux_masked = np.where(mask, ux, 0)   # 不符合条件的区域速度设为 0
        uy_masked = np.where(mask, uy, 0)   # 不符合条件的区域速度设为 0

        if vmin is not None and vmax is not None:
            color_field = np.clip(color_field, vmin, vmax)
        # 1. Plot alpha concentration cloud map (background)
        cf = plt.contourf(
            xi, yi, alpha_i,
            levels=np.linspace(0, 0.015, 128),                   # 颜色分级数
            cmap='gray_r',              # 云图颜色映射
            alpha=0.75,          # 透明度
            antialiased=True,             # 抗锯齿
            zorder=1
            
        )
        
     
         # --- 关键修改：强制颜色映射范围为 [vmin, 0.2]，即使 vmax < 0.2 ---
        # norm_stream= plt.Normalize(vmin=vmin, vmax=vmax)  # 固定颜色条上限为 0.2
        # speed = np.sqrt(ux**2 + uy**2)
        strm = plt.streamplot(
            xi, yi, ux_masked, uy_masked,
            # color='#04d8b2',
            color = '#0343df',
            # cmap='turbo',
            linewidth=1,
            density=8,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
           
        )

        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        # plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS2)

        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)
        
        plt.gca().set_aspect('auto')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(*self.X_LIM)
        plt.ylim(*self.Y_LIM)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()




      
    def plot_streamlines2(self, xi, yi, ux, uy, color_field, alpha_i, time_v, positions, y_text, 
                        title, filename, color_label, vmin=None, vmax=None):
        """Generic streamline plotting function"""
        plt.figure(figsize=self.FIG_SIZE, constrained_layout=True)
        mask = alpha_i > 1e-5  # 找出 alpha_i > 1e-5 的区域
        

        if vmin is not None and vmax is not None:
            color_field = np.clip(color_field, vmin, vmax)
        # 1. Plot alpha concentration cloud map (background)
        cf = plt.contourf(
            xi, yi, alpha_i,
            levels=np.linspace(0, 0.015, 128),                   # 颜色分级数
            cmap='gray_r',              # 云图颜色映射
            alpha=0.7,          # 透明度
            antialiased=True,             # 抗锯齿
            zorder=1
            
        )
        
     
         # --- 关键修改：强制颜色映射范围为 [vmin, 0.2]，即使 vmax < 0.2 ---
        # norm_stream= plt.Normalize(vmin=vmin, vmax=vmax)  # 固定颜色条上限为 0.2
        speed = np.sqrt(ux**2 + uy**2)
        strm = plt.streamplot(
            xi, yi, ux, uy,
            # color='#04d8b2',
            color = color_field,
            cmap='coolwarm',
            # cmap='turbo',
            linewidth=1,
            density=5,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
           
        )

        
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        # plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS2)

        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)
        
        plt.gca().set_aspect('auto')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(*self.X_LIM)
        plt.ylim(*self.Y_LIM)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_contour(self, xi, yi, field, alpha_i, time_v, positions, y_text, 
                     title, filename, color_label, levels=None, cmap='bwr'):
        """Generic contour plotting function"""
        plt.figure(figsize=self.FIG_SIZE)
        
        
        # contour = plt.contourf(
        #     xi, yi, alpha_i,
        #     levels=np.linspace(0, 0.015, 128),                   # 颜色分级数
        #     cmap='gray_r',              # 云图颜色映射
        #     alpha=1,          # 透明度
        #     antialiased=True,             # 抗锯齿
        #     zorder=1
        # )
        
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        
        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)
        
        # cbar = plt.colorbar(contour, label=color_label, orientation='horizontal',)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(*self.X_LIM)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_vorticity_contour(self, xi, yi, ux, uy, alpha_i, time_v, 
                             positions, y_text, title, filename, levels=None, 
                             cmap='bwr'):
        """
        绘制涡量云图（使用ux_perturb = ux - U 和原始uy计算涡量）
        
        参数：
            xi, yi: 网格坐标
            ux: 原始x方向速度
            uy: 原始y方向速度
            ux_perturb: 扰动速度 (ux - U)
            alpha_i: 颗粒浓度场
            time_v: 时间步
            positions: 垂直线位置字典
            y_text: 垂直线标注的y位置
            title: 图标题
            filename: 保存文件名
            levels: 云图色阶
            cmap: 颜色映射
        """
        plt.figure(figsize=self.FIG_SIZE)
        # 1. Plot alpha concentration cloud map (background)
        # cf = plt.contourf(
        #     xi, yi, alpha_i,
        #     levels=np.linspace(0, 0.015, 128),                   # 颜色分级数
        #     cmap='gray_r',              # 云图颜色映射
        #     alpha=0.7,          # 透明度
        #     antialiased=True,             # 抗锯齿
        #     zorder=1
        # )
            
        # 1. 计算涡量 (ω_z = dv/dx - du/dy)
        # 使用扰动速度计算du/dy，原始速度计算dv/dx
        dy = yi[1,0] - yi[0,0]  # y方向网格间距
        dx = xi[0,1] - xi[0,0]  # x方向网格间距
        
        # 计算速度梯度 (中心差分)
        duy = np.gradient(ux, dy, axis=0)  # ∂u'/∂y
        dvx = np.gradient(uy, dx, axis=1)          # ∂v/∂x
        
        # z方向涡量
        omega_z = dvx - duy
                # 1. Plot alpha concentration cloud map (background)
        mask = (np.abs(omega_z) > 0.001) & (alpha_i > self.alpha_threshold)
        
        # 连通区域标记（识别离散涡旋）
        from scipy.ndimage import label
        labeled, n_vortices = label(mask)
        # print(f"***Detected {n_vortices} vortices.")

        # 2. 绘制云图
        if levels is None:
            # 自动确定色阶范围（对称）
            max_val = np.nanmax(np.abs(omega_z))
            levels = np.linspace(-max_val, max_val, 41)
        
        contour = plt.contourf(
            xi, yi, omega_z,
            levels=levels,
            cmap=cmap,
            extend='both',
            alpha=0.5,
            zorder = 1
        )
        mask = alpha_i > 1e-5  # 找出 alpha_i > 1e-5 的区域
        ux_masked = np.where(mask, ux, 0)   # 不符合条件的区域速度设为 0
        uy_masked = np.where(mask, uy, 0)   # 不符合条件的区域速度设为 0


        # 1. Plot alpha concentration cloud map (background)

        
     
         # --- 关键修改：强制颜色映射范围为 [vmin, 0.2]，即使 vmax < 0.2 ---
        # norm_stream= plt.Normalize(vmin=vmin, vmax=vmax)  # 固定颜色条上限为 0.2
        # speed = np.sqrt(ux**2 + uy**2)
        strm = plt.streamplot(
            xi, yi, ux_masked, uy_masked,
            # color='#04d8b2',
            color = '#0343df',
            # cmap='turbo',
            linewidth=1,
            density=5,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
           
        )

        # 3. 添加颗粒浓度等值线
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        # plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS2)
        # 4. 添加参考线/标签
        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', 
                       linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', 
                    fontsize=20, zorder=3, color=self.colorset)
        
        # 5. 添加颜色条和其他装饰
        # cbar = plt.colorbar(contour, label='Vorticity $\omega_z$ [1/s]', 
        #                    orientation='horizontal', shrink=0.3,pad = 0.2)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(*self.X_LIM)
        plt.ylim(*self.Y_LIM)
        plt.title(title)
        
        # 6. 保存图像
        plt.savefig(os.path.join(self.output_dir, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()    




    def plot_velocity_vectors(
            self, xi, yi, ux, uy, alpha_i, time_v, positions, y_text,
        title, filename, skip_x=3, skip_y=3, vector_color='red', scale=1.0,
            arrow_scale=1.0, alpha_opacity=0.6, alpha_cmap='gray_r',velocity_zero_points=None,h_points = None, normalize_x=False, head_x=None, H0=0.3, vmin=0, vmax=0.2
        ):
        """
        Plot velocity vectors with proper arrowheads and speed-proportional lengths.
        
        Parameters:
            scale : float
                Larger values make arrows shorter (e.g., 50 = medium, 100 = short).
            skip_x, skip_y : int
                Sampling step (higher = fewer arrows).
            color : str or array-like
                Color of arrows (can be a colormap).
        """
        plt.figure(figsize=self.FIG_SIZE)
            

        x_label = 'x [m]'
        y_label = 'y [m]'


        # Mask regions where alpha < threshold (no arrows)
        alpha_mask = alpha_i > self.alpha_threshold
        masked_ux = np.where(alpha_mask, ux, np.nan)
        masked_uy = np.where(alpha_mask, uy, np.nan)
        masked_alpha = np.where(alpha_mask, alpha_i, np.nan)

        # Compute speed (magnitude of velocity)
        # 计算速度大小
        speed = np.sqrt(ux**2 + uy**2)
        
        # 归一化速度分量（保持方向，长度 = speed）
        max_speed = np.nanmax(speed)  # 避免除以零
        normalized_ux = (ux / max_speed) * arrow_scale
        normalized_uy = (uy / max_speed) * arrow_scale

        # 1. Plot alpha concentration cloud map (background)
        cf = plt.contourf(
        xi, yi, alpha_i,
        levels = np.linspace(0, 0.012, 128),                   # 颜色分级数
        cmap=alpha_cmap,              # 云图颜色映射
        alpha=alpha_opacity,          # 透明度
        antialiased=True              # 抗锯齿
        )
        cbar = plt.colorbar(cf, label=r'$\alpha_s$',orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label( label=r'$\alpha_s$',fontsize=12)
        cbar.set_ticks([0.000, 0.006, 0.012])
        cbar.ax.set_position([0.75, 0.7, 0.2, 0.02])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        # 绘制长度随 speed 变化的矢量
        q = plt.quiver(
            xi[::skip_y, ::skip_x], yi[::skip_y, ::skip_x],
            normalized_ux[::skip_y, ::skip_x],  # 归一化后的分量
            normalized_uy[::skip_y, ::skip_x],
            speed[::skip_y, ::skip_x],  # 仍用于颜色映射（可选）
            scale=scale,                # 设为 1.0 或更小的基础缩放
            scale_units='inches',       # 固定物理长度单位
            angles='xy',                # 确保方向正确
            width=0.0005 * arrow_scale,  # 箭头宽度
            headwidth=5 * arrow_scale,  # 头部宽度
            headlength=7 * arrow_scale, # 头部长度
            headaxislength=4.5 * arrow_scale,
            norm=norm,
            #color=vector_color,         # 统一颜色（或用 cmap 映射 speed）
            linewidth=0.8,
            cmap='cool' #if vector_color is None else None  # 可选颜色映射
        )
        #plt.colorbar(q, label='Velocity Magnitude')
         # 添加速度颜色图例（quiver专用colorbar）
        cbar_speed = plt.colorbar(q, label='Velocity Magnitude [m/s]',orientation='horizontal', norm=norm)
        cbar_speed.ax.tick_params(labelsize=10)
        cbar_speed.set_label('Velocity Magnitude [m/s]', fontsize=12)
        cbar_speed.set_ticks([0, vmax/2, vmax])
        cbar_speed.ax.set_position([0.75, 0.8, 0.2, 0.02])  # [left, bottom, width, height]
        

            # === 3. 添加箭头长度的参考图例 ===
        # 手动在角落添加参考箭头（需计算实际物理长度）
        ref_speed = 0.1  # 示例参考速度（可自定义）
        ref_x, ref_y = 0.9, 0.95    # 图例位置（相对坐标）
        ref_arrow = plt.quiver(
            ref_x, ref_y,
            ref_speed / max_speed * arrow_scale, 0,  # 水平箭头
            color='k', scale=3, scale_units='inches',
            width=0.0005 * arrow_scale,  # 加粗显示
            headwidth=5 * arrow_scale,
            transform=plt.gca().transAxes  # 使用相对坐标
        )
        plt.text(
            ref_x + 0.02, ref_y,
            f'{ref_speed:.2f} m/s',
            transform=plt.gca().transAxes,
            va='center', ha='left',
            fontsize=12,
        )



        

  


    def measure_vortex_dimensions(self, xi, yi, ux, uy, alpha_i, time_v):
        """
        量化每个涡旋的长宽尺寸（物理尺度）- 修改后版本
        
        参数:
            u_rot, v_rot : 旋转速度场分量（需先减去背景平均流）
            min_vorticity : 筛选显著涡旋的涡量阈值

        返回:
            vortices : list of dicts, 每个涡旋包含:
                'center' : (x,y) 涡心坐标
                'length' : 长轴长度（特征值λ1相关）
                'width'  : 短轴长度（特征值λ2相关） 
                'angle'  : 主轴角度（弧度）
                'area'   : 涡旋近似面积
        """
        ############### 1. 计算涡量场 ################
                
        dy = yi[1, 0] - yi[0, 0]
        dx = xi[0, 1] - xi[0, 0]
        dUx = np.gradient(ux, dx, axis=1)
        dUy = np.gradient(ux, dy, axis=0)
        dVx = np.gradient(uy, dx, axis=1)
        dVy = np.gradient(uy, dy, axis=0)
        Q = self.calculate_q_criterion(dUx, dUy, dVx, dVy)

        # ========== 1. 定义双阈值 ==========
        q_max = np.nanmax(Q)
        if q_max <= 0:
            return []
            
        
        q_threshold_low = 0.05    # 用来定义“焊接”的边界

    # ========== 2. 创建初始掩码 ==========
    # 这是我们严格识别出的、但可能破碎的涡核
        initial_mask = (Q > self.Q_threshold) &  (alpha_i > self.alpha_threshold)

    # ========== 3. 使用“闭运算”进行焊接 ==========
    # 关键：我们现在有了焊接的边界！闭运算只会在 Q > q_threshold_low 的区域内生效。
    # (虽然binary_closing本身不直接使用边界，但它的效果等同于此)
    
    # 结构元素的大小决定了“焊条”的长度
    # 如果断裂带很宽，就需要更大的structure
        structure = np.ones((10, 10), dtype=bool) 

    # 对初始的、破碎的掩码进行闭运算
        closed_mask = binary_closing(initial_mask, structure=structure)

    # (可选，但推荐) 确保焊接后的区域仍然在宽容的边界内
        final_mask = closed_mask & (Q > q_threshold_low)
    
    # ========== 4. 在最终处理过的掩码上标记连通区域 ==========
    # 此时，原本破碎的区域应该已经被连接成了一个整体
        # final_mask_pos = final_mask & (vorticity > 0)
        # final_mask_neg = final_mask & (vorticity < 0)

        labeled, n_vortices = label(final_mask)

        print(f"通过闭运算连接后，识别出 {n_vortices} 个涡结构。")
    
        # ========== 3. 逐个分析涡旋 ==========

        vortex_list = []  # 修改变量名避免冲突
        for i in range(1, n_vortices + 1):
            vortex_mask = (labeled == i)
            points = np.column_stack([xi[vortex_mask], yi[vortex_mask]])
            velocities = np.column_stack([ux[vortex_mask], uy[vortex_mask]])

            # ---------- PCA分析 ----------
            try:
                pca_geo = ManualPCA()
                pca_geo.fit(points)
                geo_props = pca_geo.get_vortex_properties()
                
                pca_kin = ManualPCA()
                pca_kin.fit(velocities - velocities.mean(axis=0))
                kin_props = pca_kin.get_vortex_properties()
            except Exception as e:
                print(f"涡旋 {i} PCA分析失败：{e}")
                continue

            # --- FIX STARTS HERE ---
            vortex_list.append({
                'id': i,
                'time': time_v,  # 使用传入的时间参数
                'center': geo_props['center'],
                'length': geo_props['length'],
                'width': geo_props['width'],
                'geo_angle_deg': np.degrees(geo_props['angle']),
                'kin_angle_deg': np.degrees(kin_props.get('angle', 0)),
                'area': np.pi * (geo_props['length']/2) * (geo_props['width']/2),
                'max_Q_value': np.max(Q[vortex_mask]),
                'geo_major_axis': geo_props['major_axis'], # Store the major axis vector
                'kin_major_axis': kin_props['major_axis']  # Store the major axis vector
            })
            # --- FIX ENDS HERE ---
        
        # 保存数据
        if vortex_list:  # 仅当有数据时保存
            self.save_vortex_data(vortex_list)
        
        return vortex_list

    
    def save_vortex_data(self, vortex_list):
        """保存涡旋数据到CSV文件（使用pandas）"""
        import pandas as pd
        
        # 准备数据字典
        data_to_save = []
        for vortex in vortex_list:
            data_to_save.append({
                'id': vortex['id'],
                'time': vortex['time'],
                'x_center': vortex['center'][0],
                'y_center': vortex['center'][1],
                'length': vortex['length'],
                'width': vortex['width'],
                'geo_angle_deg': vortex['geo_angle_deg'],
                'kin_angle_deg': vortex['kin_angle_deg'],
                'area': vortex['area'],
                'max_Q_value': vortex['max_Q_value']
            })
        
        df = pd.DataFrame(data_to_save)
        filepath = os.path.join(self.output_dir, self.vortex_output_file)
        
        # 追加或新建文件
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        print(f"已保存 {len(df)} 条涡旋数据到 {filepath}")

    
    def plot_vortex_boundaries(self, xi, yi, vortices, u_rot, v_rot, time_v, alpha_i,positions,y_text):
        """在速度场上叠加显示PCA测量的涡旋椭圆（改进版）"""
        plt.figure(figsize=self.FIG_SIZE)

        # 1. Plot alpha concentration cloud map (background)
        cf = plt.contourf(
            xi, yi, alpha_i,
            levels=np.linspace(0, 0.015, 128),                   # 颜色分级数
            cmap='gray_r',              # 云图颜色映射
            alpha=0.75,          # 透明度
            antialiased=True,             # 抗锯齿
            zorder=1
            
        )
        
        # 1. 绘制背景流线
        speed = np.sqrt(u_rot**2 + v_rot**2)
        mask = alpha_i > 1e-5  # 找出 alpha_i > 1e-5 的区域
        ux_masked = np.where(mask, u_rot, 0)   # 不符合条件的区域速度设为 0
        uy_masked = np.where(mask, v_rot, 0)   # 不符合条件的区域速度设为 0
    
        plt.streamplot(
            xi, yi, ux_masked, uy_masked,
            # color='#04d8b2',
            color = '#0343df',
            # cmap='turbo',
            linewidth=1,
            density=8,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
        )
        # self.plot_streamlines(xi, yi, u_rot, v_rot, speed, alpha_i, time_v, positions, y_text, 
        #                 title='', filename='', color_label='', vmin=0, vmax=0.2)
        # 2. 绘制每个涡旋的几何椭圆和主轴
        
        # --- FIX STARTS HERE ---
        # Changed loop variable from 'vortex_list' to 'vortex' for clarity
        for vortex in vortices:
            # 几何形状椭圆
            ellipse = Ellipse(
                xy=vortex['center'],
                width=vortex['length'],
                height=vortex['width'],
                angle=vortex['geo_angle_deg'],
                edgecolor='r',
                facecolor='none',
                linestyle='--',
                linewidth=2,
                zorder=3
            )
            plt.gca().add_patch(ellipse)
                        
            # 几何主轴 (使用已存储的单位向量)
            p1 = vortex['center'] - 0.5 * vortex['length'] * vortex['geo_major_axis']
            p2 = vortex['center'] + 0.5 * vortex['length'] * vortex['geo_major_axis']
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1.5, zorder=4)
                
            
            
            # 动力学主轴（用蓝色表示）
            plt.plot(
                [vortex['center'][0] - 0.5*vortex['length']*vortex['kin_major_axis'][0],
                 vortex['center'][0] + 0.5*vortex['length']*vortex['kin_major_axis'][0]],
                [vortex['center'][1] - 0.5*vortex['length']*vortex['kin_major_axis'][1],
                 vortex['center'][1] + 0.5*vortex['length']*vortex['kin_major_axis'][1]],
                color='magenta', linestyle='--',linewidth=1.5, zorder=4
            )
            
            # 标记中心点和编号
            plt.scatter(*vortex['center'], c='r', s=50, marker='x', zorder=5)
            # Use vortex['id'] instead of the loop index 'i'
            plt.text(vortex['center'][0], vortex['center'][1], str(vortex['id']),
                    color='k', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6)
        # --- FIX ENDS HERE ---
            
            # plot alpha contours again to be on top
            plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)

            for label, x_pos in positions.items():
                plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
                plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)
        
        # 图例和装饰
        plt.title(f'Vortex Boundaries (PCA Analysis) at t={time_v}s')
        plt.xlim(self.X_LIM)
        plt.ylim(0, 0.3)
        
        # 手动创建图例元素
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='r', linestyle='-', lw=2, label='Geometric Axis'),
            Line2D([0], [0], color='magenta', linestyle='--', lw=2, label='Kinematic Axis'),
            Line2D([0], [0], marker='x', color='r', lw=0, label='Vortex Center')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(os.path.join(self.output_dir, f'vortex_pca_dimensions_t{time_v}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    

    def calculate_lambda2_criterion(self,ux, uy, xi, yi, sigma=1.0, lambda2_threshold=0):
        """
        计算 λ₂ 准则并生成涡旋掩膜。
        
        参数:
            ux, uy      : 速度场分量 (2D数组)
            xi, yi      : 网格坐标 (2D数组)
            sigma       : 高斯滤波系数（去噪）
            lambda2_threshold : λ₂ 判定阈值（一般取负值，如-0.1）
        
        返回:
            lambda2     : λ₂ 值场 (2D数组)
            vortex_mask : 涡旋区域二值掩膜 (True/False)
        """
        # 1. 高斯滤波去噪（可选）
        ux = gaussian_filter(ux, sigma=sigma)
        uy = gaussian_filter(uy, sigma=sigma)
        ux = np.nan_to_num(ux)
        uy = np.nan_to_num(uy)
        # 2. 计算速度梯度 (du/dx, du/dy, dv/dx, dv/dy)
        dy = yi[1, 0] - yi[0, 0]
        dx = xi[0, 1] - xi[0, 0]
        dudx = np.gradient(ux, dx,axis=1)  # ∂u/∂x
        dudy = np.gradient(ux, dy,axis=0)  # ∂u/∂y
        dvdx = np.gradient(uy, dx,axis=1)  # ∂v/∂x
        dvdy = np.gradient(uy, dy,axis=0)  # ∂v/∂y
        
        # 3. 构造速度梯度张量 ∇v 并分解为 S 和 Ω
        S = np.zeros((*ux.shape, 2, 2))
        Omega = np.zeros_like(S)
        
        S[..., 0, 0] = dudx                     # S11
        S[..., 0, 1] = 0.5 * (dudy + dvdx)      # S12 = S21
        S[..., 1, 0] = S[..., 0, 1]
        S[..., 1, 1] = dvdy                     # S22
        
        Omega[..., 0, 1] = 0.5 * (dudy - dvdx)  # Ω12
        Omega[..., 1, 0] = -Omega[..., 0, 1]    # Ω21
        
        # 4. 计算 S² + Ω² 的特征值 λ₂
        lambda2 = np.zeros_like(ux)
        for i in range(ux.shape[0]):
            for j in range(ux.shape[1]):
                M = S[i, j] @ S[i, j] + Omega[i, j] @ Omega[i, j]
                eigvals = np.linalg.eigvals(M)
                lambda2[i, j] = np.sort(eigvals)[1]  # 取第二大的特征值（λ₂）
        
        # 5. 生成涡旋掩膜：λ₂ < threshold
        vortex_mask = lambda2 < self.lambda2_threshold
        
        return lambda2, vortex_mask

    def plot_lambda2_results(self,xi, yi, alpha_i,lambda2, vortex_mask, ux=None, uy=None, time_v=0):
        """
        可视化 λ₂ 准则结果。
        """
        plt.figure(figsize=self.FIG_SIZE)
        
        # # 1. 绘制 λ₂ 场
        # plt.subplot(1, 2, 1)
        # plt.contourf(xi, yi, lambda2, levels=50, cmap='RdBu')
        # plt.colorbar(label='λ₂')
        # plt.title('λ₂ Field')
        
        # 2. 绘制涡旋掩膜（叠加流线）
        # plt.subplot(1, 2, 2)
        plt.contourf(xi, yi, vortex_mask, levels=[0, 0.5, 1], colors=['none', 'red'], alpha=0.3)
                
        mask = alpha_i > 1e-5  # 找出 alpha_i > 1e-5 的区域
        ux_masked = np.where(mask, ux, 0)   # 不符合条件的区域速度设为 0
        uy_masked = np.where(mask, uy, 0)   # 不符合条件的区域速度设为 0
    
        plt.streamplot(
            xi, yi, ux_masked, uy_masked,
            # color='#04d8b2',
            color = '#0343df',
            # cmap='turbo',
            linewidth=1,
            density=8,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
        )
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)

        plt.title('Vortex Regions (λ₂ < 0)')
        plt.xlim(self.X_LIM)
        plt.ylim(0, 0.3)
        
        plt.savefig(os.path.join(self.output_dir, f'vortex_lamda_dimensions_t{time_v}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


    def measure_vortex_dimensions_lambda2(self, xi, yi, ux, uy, alpha_i, time_v):
        """
        基于Lambda2准则量化涡旋特征（按x坐标排序）
        """
        # 1. 计算Lambda2场
        lambda2, _ = self.calculate_lambda2_criterion(ux, uy, xi, yi, lambda2_threshold=0)
        
        # 2. 创建初始掩码 (Lambda2 < 0 且 alpha > threshold)
        initial_mask = (lambda2 < self.lambda2_threshold) & (alpha_i > self.alpha_threshold)
        
        # 3. 使用"闭运算"连接破碎区域
        structure = np.ones((10, 10), dtype=bool)
        closed_mask = binary_closing(initial_mask, structure=structure)
        
        # 4. 最终掩码 (确保仍然满足Lambda2 < 0)
        final_mask = closed_mask & (lambda2 < 0)
        
        # 5. 标记连通区域
        labeled, n_vortices = label(final_mask)
        print(f"通过Lambda2准则识别出 {n_vortices} 个涡结构")
        
        # 6. 按x坐标重新排序涡旋ID
        vortex_data = []  # 存储(旧ID, 中心x坐标, PCA属性)
        
        # 先收集所有涡旋的数据
        for i in range(1, n_vortices + 1):
            vortex_mask = (labeled == i)
            points = np.column_stack([xi[vortex_mask], yi[vortex_mask]])
            
            try:
                pca = ManualPCA()
                pca.fit(points)
                props = pca.get_vortex_properties()
                center_x = props['center'][0]  # 获取x坐标
                area = np.pi * (props['length']/2) * (props['width']/2)
                if area > 1e-4:  # 面积过滤条件
                    
                    vortex_data.append({
                    'old_id': i,
                    'center_x': center_x,
                    'props': props,
                    'vortex_mask': vortex_mask
                })
                else:
                    print(f"涡旋 原ID{i} 面积过小 ({area:.6f})，已过滤")
                
            except Exception as e:
                print(f"涡旋 {i} PCA分析失败：{e}")
                continue
        
        # 按x坐标从小到大排序
        vortex_data.sort(key=lambda x: x['center_x'], reverse=True)
        
        print(f"重新排序后：{[(idx+1, data['old_id'], data['center_x']) for idx, data in enumerate(vortex_data)]}")
        
        # 7. 使用新ID构建最终的涡旋列表
        vortex_list = []
        for idx, data in enumerate(vortex_data):
            new_id = idx + 1
            old_id = data['old_id']
            props = data['props']
            vortex_mask = data['vortex_mask']
            
            # 计算区域内最大Lambda2值
            max_lambda2 = np.min(lambda2[vortex_mask])  # Lambda2是负值，取最小即绝对值最大
            area = np.pi * (props['length']/2) * (props['width']/2)
            
            # 面积判断和过滤逻辑
            
            vortex_list.append({
                    'id': new_id,  # 使用新的排序ID
                    'time': time_v,
                    'center': props['center'],
                    'length': props['length'],
                    'width': props['width'],
                    'angle': np.degrees(props['angle']),
                    'area': area,
                    'max_lambda2': max_lambda2,
                    'major_axis': props['major_axis'],
                    'original_id': old_id  # 可选：保留原始ID用于调试
                })
            
        
        # 保存数据
        if vortex_list:
            self.save_vortex_data_lambda2(vortex_list)
        
        return vortex_list



    def save_vortex_data_lambda2(self, vortex_list):
        """保存Lambda2涡旋数据到CSV"""
        import pandas as pd
        
        data = []
        for vortex in vortex_list:
            data.append({
                'id': vortex['id'],
                'time': vortex['time'],
                'x_center': vortex['center'][0],
                'y_center': vortex['center'][1],
                'length': vortex['length'],
                'width': vortex['width'],
                'angle': vortex['angle'],
                'area': vortex['area'],
                'max_lambda2': vortex['max_lambda2']
            })
        
        df = pd.DataFrame(data)
        filename = f"lambda2_vortex_properties_{self.timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        print(f"保存 {len(df)} 条Lambda2涡旋数据到 {filepath}")

    def plot_lambda2_vortex_boundaries(self, xi, yi, vortices, ux, uy, alpha_i, time_v,positions,y_text):
        """
        可视化Lambda2识别的涡旋边界
        """
        plt.figure(figsize=self.FIG_SIZE)
        # 1. Plot alpha concentration cloud map (background)
        cf = plt.contourf(
            xi, yi, alpha_i,
            levels=np.linspace(0, 0.015, 128),                   # 颜色分级数
            cmap='gray_r',              # 云图颜色映射
            alpha=0.75,          # 透明度
            antialiased=True,             # 抗锯齿
            zorder=1
            
        )
        # === 添加精细网格 ===
        # 设置网格线的密度
        x_ticks = np.arange(self.X_LIM[0], self.X_LIM[1] + 0.1, 0.1)  # 每0.1m一条竖线
        y_ticks = np.arange(self.Y_LIM[0], self.Y_LIM[1] + 0.05, 0.05)  # 每0.05m一条横线
        
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.grid(True, alpha=1, linestyle='-', linewidth=0.5, color='lightgray')
        
        # 1. 背景流线
        mask = alpha_i > 1e-5
        ux_masked = np.where(mask, ux, 0)
        uy_masked = np.where(mask, uy, 0)
        
        plt.streamplot(
            xi, yi, ux_masked, uy_masked,
            color='#0343df',
            linewidth=1,
            density=8,
            arrowsize=2,
            arrowstyle='->',
            zorder=2
        )
        
        # 2. 绘制每个涡旋的椭圆
        for vortex in vortices:
            ellipse = Ellipse(
                xy=vortex['center'],
                width=vortex['length'],
                height=vortex['width'],
                angle=vortex['angle'],
                edgecolor='r',
                facecolor='none',
                linestyle='--',
                linewidth=2,
                zorder=3
            )
            plt.gca().add_patch(ellipse)
            
            # 绘制主轴
            p1 = vortex['center'] - 0.5*vortex['length']*vortex['major_axis']
            p2 = vortex['center'] + 0.5*vortex['length']*vortex['major_axis']
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1.5, zorder=4)
            
            # 标记中心点和ID
            plt.scatter(*vortex['center'], c='r', s=50, marker='x', zorder=5)
            plt.text(vortex['center'][0], vortex['center'][1], str(vortex['id']),
                    color='k', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6)
        
        # 3. 添加alpha等值线和其他装饰
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        for label, x_pos in positions.items():
                plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
                plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)
        plt.title(f'Lambda2 Vortex Boundaries (t={time_v}s)')
        plt.xlim(self.X_LIM)
        plt.ylim(self.Y_LIM)
        
        # 保存图像
        filename = f'lambda2_vortex_boundaries_t{time_v}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
   


    def process_time_step(self, X,Y,Z,time_v):
        """Process data for a single time step"""

        select = (Z == 0.135)

        # Read field data
        Ua_A = fluidfoam.readvector(self.sol, str(time_v), "U.a")
        alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
        beta = fluidfoam.readscalar(self.sol, str(time_v), "alpha.b")
        gradU = fluidfoam.readtensor(self.sol, str(time_v), "grad(U.a)")
        vorticity = fluidfoam.readvector(self.sol, str(time_v), "vorticity")
        gradbeta = fluidfoam.readvector(self.sol, str(time_v), "grad(alpha.b)")
        gradvorticity = fluidfoam.readtensor(self.sol, str(time_v), "grad(vorticity)")

        # selecting slice at Z = xxx
        X = X[select]
        Y = Y[select]
        alpha_A = alpha_A[select]
        Ua_A = Ua_A[:,select]
        beta = beta[select]
        gradU = gradU[:,select]
        vorticity = vorticity[:,select]
        gradbeta = gradbeta[:,select]
        gradvorticity = gradvorticity[:,select]
        # Extract components
        gradU_x = gradU[0]
        gradU_y = gradU[3]
        gradV_x = gradU[1]
        gradV_y = gradU[4]
        omega_z = vorticity[2]
        gradbeta_x = gradbeta[0]
        gradvorticity_x = gradvorticity[2]
        gradvorticity_y = gradvorticity[5]

        velocity_zero_points = []
        h_points = []

        # Locate head position

        head_x = None
        for x in np.unique(X):
            mask = (X == x) & (Y >= self.y_min) & (alpha_A > self.alpha_threshold)
            if np.any(mask):
                head_x = x
        if head_x is None:
            print(f"Warning: No head found at t={time_v}")
            return

        # Process each x coordinate
        results = []
        x_coords = np.unique(X[(X <= head_x) & (X >= 0)])
        
        for xx in x_coords:
            mask = (X == xx)  & (Y >= 0) & (alpha_A > 1e-5)
            if not np.any(mask):
                continue
            
            ya = Y[mask]
            ua = Ua_A[0][mask]
            alpha = np.maximum(alpha_A[mask], 0)
            
            # Sort by y
            # sort_idx = np.argsort(ya)
            ya = ya
            ua_x = ua
            alpha_vals = alpha

            # Calculate quantities
            U, H, ALPHA, H_depth, y_crossing = self.integrate_quantities(ya, ua_x, alpha_vals)
            
            # Additional calculations for alpha crossing
            y_threshold = 0
            valid_mask = ya > y_threshold
            if np.any(valid_mask):
                below_threshold = (alpha_vals[valid_mask] < 1e-5)
                if np.any(below_threshold):
                    first_below_rel_index = np.argmax(below_threshold)
                    valid_indices = np.where(valid_mask)[0]
                    max_ya_crossing_index_alpha = valid_indices[first_below_rel_index]
                    y_crossing_alpha = ya[max_ya_crossing_index_alpha]
                    u_crossing_alpha = ua_x[max_ya_crossing_index_alpha]
                else:
                    max_ya_crossing_index_alpha = np.where(valid_mask)[0][-1]
                    y_crossing_alpha = ya[max_ya_crossing_index_alpha]
                    u_crossing_alpha = ua_x[max_ya_crossing_index_alpha]
            else:
                max_ya_crossing_index_alpha = len(ya) - 1
                y_crossing_alpha = ya[max_ya_crossing_index_alpha]
                u_crossing_alpha = ua_x[max_ya_crossing_index_alpha]

            # Additional integrations for alpha crossing
            ua_alpha_alpha = ua_x * alpha_vals
            integral_alpha = np.trapz(ua_alpha_alpha[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha])
            integralU_alpha = np.trapz(ua_x[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha])
            integralU2_alpha = np.trapz(ua_x[:max_ya_crossing_index_alpha]**2, ya[:max_ya_crossing_index_alpha])
            integral2_alpha = np.trapz((ua_x[:max_ya_crossing_index_alpha] * alpha_vals[:max_ya_crossing_index_alpha])**2, ya[:max_ya_crossing_index_alpha])
            
            U_alpha = integralU2_alpha / integralU_alpha if integralU_alpha != 0 else 0
            H_alpha = integral_alpha**2 / integral2_alpha if integral2_alpha != 0 else 0
            ALPHA_alpha = integral_alpha / integralU_alpha if integralU_alpha != 0 else 0

            results.append({
                "Time": time_v,
                "x": xx,
                "U": U,
                "H": H_depth,
                "y_crossing": y_crossing,
                "U_alpha": U_alpha,
                "H_alpha": H_alpha,
                "ALPHA_alpha": ALPHA_alpha,
                "y_crossing_alpha": y_crossing_alpha
            })
            # 存储速度零点的 (x, y)
            velocity_zero_points.append((xx, y_crossing))
            h_points.append((xx, H_depth))


        # Save results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, f"integration_results_t{time_v}.csv"), index=False)
        print(f"Results saved for t={time_v} (head_x={head_x:.2f}m)")

        # Calculate perturbation fields
        x_U_mapping = dict(zip(df['x'], df['U']))
        x_U_mapping_alpha = dict(zip(df['x'], df['U_alpha']))
        
        fields = self.calculate_perturbation_fields(
            X, Y, Ua_A, x_coords, x_U_mapping, x_U_mapping_alpha, gradbeta_x, omega_z, gradvorticity_x, beta
        )
        U_perturb, U_perturb_alpha, Umean_densitygradient, Uper_densitygradient, Umean_advection, Uper_advection, Uori_advection = fields

        # Interpolate fields
        xi = np.linspace(X.min(), X.max(), 750)#1000
        yi = np.linspace(Y.min(), Y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)
        
        interpolated = self.interpolate_fields(
            X, Y, xi, yi, {
                'U_perturb': U_perturb,
                'U_perturb_alpha': U_perturb_alpha,
                'Umean_densitygradient': Umean_densitygradient,
                'Uper_densitygradient': Uper_densitygradient,
                'Umean_advection': Umean_advection,
                'Uper_advection': Uper_advection,
                'Uori_advection': Uori_advection,
                'uxi': Ua_A[0],
                'uyi': Ua_A[1],
                'alpha_i': alpha_A,
                'omega_z': omega_z,
                'Q': self.calculate_q_criterion(gradU_x, gradU_y, gradV_x, gradV_y),

            }
        )


        # Define positions for vertical lines
        positions = {
            
            '$0.25H_0$': head_x - 0.25*self.Height,
            '$0.3H_0$': head_x - 0.33*self.Height,
            '$0.5H_0$': head_x - 0.5*self.Height,
            '$H_0$': head_x - self.Height
        }
        y_text = 0.32

        self.plot_vorticity_contour(
            xi, yi, 
            interpolated['U_perturb'],         # 原始x方向速度
            interpolated['uyi'],         # 原始y方向速度
            interpolated['alpha_i'],     # 颗粒浓度
            time_v,
            positions, 
            y_text,
            title=f'Vorticity Field ( $\hat U_s$ [m/s], t={time_v}s)',
            filename=f'Vorticity_t{time_v}.png',
            levels=np.linspace(-5, 5, 41),  # 自定义色阶范围
            cmap='bwr'                       # 红蓝配色
        )
                # 在process_time_step方法中（计算PCA前）
        v_rot = interpolated['uyi']  # 旋转速度y分量
        u_rot = interpolated['U_perturb']  # 旋转速度x分量

        # 测量所有涡旋尺寸
        vortex_properties = self.measure_vortex_dimensions(
            xi, yi,u_rot, v_rot, 
            interpolated['alpha_i'],time_v
              # 根据实际情况调整阈值
        )
        vortex_lamda_properties = self.measure_vortex_dimensions_lambda2(
            xi, yi,u_rot, v_rot, 
            interpolated['alpha_i'],time_v
              # 根据实际情况调整阈值
        )

        lambda2, vortex_mask = self.calculate_lambda2_criterion(
        u_rot, v_rot, xi, yi, 
                 # 高斯滤波系数
        lambda2_threshold=0 # 判定阈值（λ₂ < 0）
    )
        
        self.plot_lambda2_results(xi, yi, interpolated['alpha_i'],lambda2, vortex_mask, u_rot, v_rot,time_v)




    
    # 打印结果
        # print("\n=== 涡旋特性汇总 ===")
        # print(f"{'ID':<4} | {'X中心':<8} | {'Y中心':<8} | {'长度(m)':<8} | {'宽度(m)':<8} | {'面积(m²)':<8}")
        # print("-"*70)
        
        # for vortex in vortex_properties:
        #     print(f"{vortex['id']:<4} | {vortex['x_center']:<8.3f} | {vortex['y_center']:<8.3f} | "
        #         f"{vortex['length']:<8.3f} | {vortex['width']:<8.3f} | {vortex['area']:<8.3f}")


        # 可视化标记涡旋边界（可选）
        self.plot_vortex_boundaries(
            xi, yi, 
            vortex_properties, 
            u_rot, v_rot,
            time_v,
            interpolated['alpha_i'],
            positions,
            y_text
        )
        self.plot_lambda2_vortex_boundaries(
            xi, yi, 
            vortex_lamda_properties, 
            u_rot, v_rot,
            interpolated['alpha_i'],
            time_v,
            positions,
            y_text
        )

        # self.plot_Q_contour(
        #     xi, yi,vortex_properties,
        #     interpolated['U_perturb'],         # 原始x方向速度
        #     interpolated['uyi'],         # 原始y方向速度
        #     interpolated['alpha_i'],     # 颗粒浓度
        #     time_v,
        #     positions,
        #     y_text,
        #     title=f'Q-Criterion Contours (t={time_v}s)',
        #     filename=f'Q_contours_t{time_v}.png',
        #     levels=np.linspace(-5, 5, 21),
        #     cmap='bwr'
        # )





        # Generate plots
        self.plot_streamlines(
            xi, yi, interpolated['U_perturb'], interpolated['uyi'], 
            interpolated['U_perturb'], interpolated['alpha_i'], time_v, 
            positions, y_text, f'Rotation Velocity Streamlines (t={time_v})',
            f'Rotation_streamlines_t{time_v}s.png', 
            "$\hat{U}_s$ [m/s]", -0.3, 0.07
        )
        self.plot_streamlines(
            xi, yi, interpolated['uxi'], interpolated['uyi'], 
            interpolated['uxi'], interpolated['alpha_i'], time_v, 
            positions, y_text, f'Original Velocity Streamlines (t={time_v})',
            f'Original_streamlines_t{time_v}s.png', 
             "${U}_s$ [m/s]", -0.3, 0.07
        )    






    def run_analysis(self):
        """Main method to run the analysis for all time steps"""
        os.makedirs(self.output_dir, exist_ok=True)
        X, Y, Z = fluidfoam.readmesh(self.sol)
        for time_v in self.times:
            print(f"Processing time step: {time_v}")
            self.process_time_step(X,Y,Z,time_v)
        
        print(f"All results saved to: {self.output_dir}")

class ManualPCA:

    """
    手动实现的PCA分析，用于涡旋几何特征量化
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None     # 主成分方向 (按方差排序)
        self.explained_variance_ = None  # 各主成分的方差(特征值)
        self.mean_ = None           # 数据均值(质心)

    def fit(self, X):
        """
        计算PCA的主成分
        
        参数:
            X : np.ndarray, shape (n_samples, n_features)
                输入数据矩阵，每行一个样本，每列一个特征（这里是坐标或速度）
        """
        # Step 1: 数据去中心化
        self.mean_ = np.mean(X, axis=0)

        
        # Step 2: 计算协方差矩阵
        # rowvar=False表示特征在列上
        cov_matrix = np.cov(X, rowvar=False)
        
        # Step 3: 特征分解
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        
        # Step 4: 按特征值降序排序
        sort_indices = np.argsort(eigvals)[::-1]
        self.components_ = eigvecs[:, sort_indices]
        self.explained_variance_ = eigvals[sort_indices]
        
        return self

    def get_vortex_properties(self):
        """
        返回涡旋的几何特征
        
        返回:
            dict: 包含涡旋长、宽、角度等信息的字典
        """
        # 确保已经进行了fit
        if self.components_ is None:
            raise ValueError("请先调用fit()方法")
            
        # 主成分的长度比例
        length = 4 * np.sqrt(self.explained_variance_[0])  # 4σ长度
        width = 4 * np.sqrt(self.explained_variance_[1])   # 4σ宽度
        
        # 主轴单位向量及其角度
        major_axis = self.components_[:, 0]
        # 确保主轴方向与速度场一致（避免180°歧义）
        if major_axis[0] < 0:  # 如果x分量为负，翻转方向
            major_axis *= -1
        
        angle = np.arctan2(major_axis[1], major_axis[0])  # 弧度
        
        return {
            'center': self.mean_,
            'length': length,
            'width': width,
            'angle': angle,
            'major_axis': major_axis,
            'minor_axis': self.components_[:, 1]
        }
    

if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()


