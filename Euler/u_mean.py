import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA


class TurbidityCurrentAnalyzer:
    def __init__(self):
        # Configuration parameters
        self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
        # self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090912_1"
        self.output_dir = "/home/amber/postpro/u_umean_tc"
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.times = [10]
        self.FIG_SIZE = (40, 6)
        self.X_LIM = (0.0, 1.6)
        self.Y_LIM = (0.0, 0.3)
        self.Height = 0.3
        self.colorset = 'fuchsia'
        
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
        
        return 0.5*(Omega_norm - S_norm)

    def signed_smooth(self, Q, omega_z, sigma=1.5):
        """Smooth while preserving sign"""
        sign = np.sign(omega_z)
        magnitude = np.abs(omega_z)
        magnitude_smooth = gaussian_filter(magnitude, sigma=sigma)
        return np.where(Q > 0, sign * magnitude_smooth, np.nan)

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
            density=5,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
           
        )



        # cbar = plt.colorbar(strm.lines, label=color_label, orientation='horizontal', fraction=0.045,aspect = 30, pad=0.2)
                # --- 3. 双颜色条（水平对齐）---
        # (A) Contourf colorbar (左侧)
        # cbar_alpha = plt.colorbar(
        #     cf,
        #     label=r'$\alpha_s$',
        #     orientation='horizontal',
        #     pad=0.15,   # 调整间距（与stream bar一致）
        #     aspect=30,
        #     shrink=0.2,  # 缩小宽度以适应左侧
        # )
        # 手动调整 contounf 颜色条位置 [左, 下, 宽, 高]
        #cbar_alpha.ax.set_position([0.20, 0.35, 0.30, 0.03])  # 左对齐，高度与stream bar一致
        # cbar_alpha.set_ticks([0.000, 0.0075, 0.015])

        # (B) Streamplot colorbar (右侧)
        # cbar_speed = plt.colorbar(
        #     strm.lines,
        #     label='Velocity [m/s]',
        #     orientation='horizontal',
        #     pad=0.15,    # 保持原位置
        #     aspect=30,
        #     shrink=0.45,  # 缩小宽度以适应右侧
        # )
        # 手动调整 stream 颜色条位置 [左, 下, 宽, 高]
        #cbar_speed.ax.set_position([0.50, 0.35, 0.30, 0.03])  # 右对齐，高度与alpha bar一致
        #cbar_speed.set_ticks([vmin, (vmin+vmax)/2, vmax])
        #cbar.ax.tick_params(labelsize=10)

        #cbar.set_label( label=r'$\alpha_s$',fontsize=12)
        
        plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)

        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)
        

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



        # 3. Overlay alpha contour
        c=plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        

        # # 4. Add reference lines/labels
        # for label, x_pos in positions.items():
        #     plt.axvline(x=x_pos, color='b', linestyle=':', linewidth=0.8)
        #     plt.text(x_pos + 0.01, y_text, label, fontsize=12, color='b')


            # === 新增：绘制速度零点连线 ===
        if velocity_zero_points and len(velocity_zero_points) > 0:
            zero_x, zero_y = zip(*velocity_zero_points)
            
            # 按 x 坐标排序以确保连线顺序正确
            sorted_indices = np.argsort(zero_x)
            zero_x_sorted = np.array(zero_x)[sorted_indices]
            zero_y_sorted = np.array(zero_y)[sorted_indices]
            
            # 绘制虚线
            plt.plot(
                zero_x_sorted, zero_y_sorted,
                color='red',               # 蓝色虚线（可自定义）
                linestyle='--',
                linewidth=2,
                label='Velocity Zero Crossing',
                zorder=4                    # 确保在箭头上方
            )    


            # === 新增：绘制积分H连线 ===
        if h_points and len(h_points) > 0:
            h_x, h_y = zip(*h_points)
            
            # 按 x 坐标排序以确保连线顺序正确
            sorted_indices = np.argsort(h_x)
            h_x_sorted = np.array(h_x)[sorted_indices]
            h_y_sorted = np.array(h_y)[sorted_indices]
            
            # 绘制虚线
            plt.plot(
                h_x_sorted, h_y_sorted,
                color='purple',               # 蓝色虚线（可自定义）
                linestyle='--',
                linewidth=2,
                label='Velocity Zero Crossing',
                zorder=4                    # 确保在箭头上方
            ) 



        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(*self.X_LIM if not normalize_x else (0,self.X_LIM[1]-head_x)/H0)
        plt.ylim(0,0.3)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_velocity_vectors2(
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



        # 3. Overlay alpha contour
        c=plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        

        # 4. Add reference lines/labels
        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)


        #     # === 新增：绘制速度零点连线 ===
        # if velocity_zero_points and len(velocity_zero_points) > 0:
        #     zero_x, zero_y = zip(*velocity_zero_points)
            
        #     # 按 x 坐标排序以确保连线顺序正确
        #     sorted_indices = np.argsort(zero_x)
        #     zero_x_sorted = np.array(zero_x)[sorted_indices]
        #     zero_y_sorted = np.array(zero_y)[sorted_indices]
            
        #     # 绘制虚线
        #     plt.plot(
        #         zero_x_sorted, zero_y_sorted,
        #         color='red',               # 蓝色虚线（可自定义）
        #         linestyle='--',
        #         linewidth=2,
        #         label='Velocity Zero Crossing',
        #         zorder=4                    # 确保在箭头上方
        #     )    


            # === 新增：绘制积分H连线 ===
        if h_points and len(h_points) > 0:
            h_x, h_y = zip(*h_points)
            
            # 按 x 坐标排序以确保连线顺序正确
            sorted_indices = np.argsort(h_x)
            h_x_sorted = np.array(h_x)[sorted_indices]
            h_y_sorted = np.array(h_y)[sorted_indices]
            
            # 绘制虚线
            plt.plot(
                h_x_sorted, h_y_sorted,
                color='red',               # 蓝色虚线（可自定义）
                linestyle='--',
                linewidth=2,
                label='Velocity Zero Crossing',
                zorder=4                    # 确保在箭头上方
            ) 



        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(*self.X_LIM if not normalize_x else (0,self.X_LIM[1]-head_x)/H0)
        plt.ylim(0,0.3)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_velocity_vectors3(
            self, xi, yi, ux, uy, alpha_i, time_v, positions, y_text,
        title, filename, skip_x=3, skip_y=3, vector_color='red', scale=1.0,
            arrow_scale=1.0, alpha_opacity=0.6, alpha_cmap='gray_r',velocity_zero_points=None,h_points = None, normalize_x=False, head_x=None, H0=0.3, vmin=0, vmax=0.2,Hi=None
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
        # Compute speed (magnitude of velocity)
        # 计算速度大小
        # 保持原始网格结构计算速度
        speed = np.sqrt(ux**2 + uy**2)  # 依然是二维数组

# Create mask for valid regions (alpha > 1e-5 and 0.005 < y < Hi)
        alpha_mask =  (yi > 0.005) 
        # 分割高速度区(y>Hi)和关注区(y≤Hi)
        mask_high_speed = (yi > Hi)   # y>Hi的区域
        mask_focus_area = (yi <= Hi)  & (yi > 0.05)# 你关注的y≤Hi区域

        # 分别计算两区的最大速度
        max_speed_high = np.nanmax(np.where(mask_high_speed, speed, np.nan))
        max_speed_focus = np.nanmax(np.where(mask_focus_area, speed, np.nan))

        # 如果关注区速度太小，限制最小缩放比例
        if max_speed_focus < 0.1 * max_speed_high:
            max_speed_focus = max_speed_high * 0.1



        # 用掩码区域计算最大速度
        # 高速度区：缩小箭头
        ux_normalized = np.where(
            mask_high_speed, 
            ux * (arrow_scale * 0.05) / max_speed_high,  # 缩小30%
            ux * arrow_scale*10 / max_speed_focus  # 关注区正常缩放
        )

        uy_normalized = np.where(
            mask_high_speed,
            uy * (arrow_scale * 0.05) / max_speed_high,
            uy * arrow_scale *10/ max_speed_focus
        )


        # # 归一化+掩码一步完成
        # normalized_ux = np.where(alpha_mask, (ux/max_speed)*arrow_scale, np.nan)
        # normalized_uy = np.where(alpha_mask, (uy/max_speed)*arrow_scale, np.nan)



        
        
        
        # # Apply mask to velocity components
        # masked_ux = np.where(alpha_mask, normalized_ux, np.nan)
        # masked_uy = np.where(alpha_mask, normalized_uy, np.nan)
        # masked_speed = np.where(alpha_mask, speed, np.nan)

        # Plot alpha concentration background
        cf = plt.contourf(
            xi, yi, alpha_i,
            levels=np.linspace(0, 0.012, 128),
            cmap=alpha_cmap,
            alpha=alpha_opacity,
            antialiased=True
        )
        
        # Add alpha colorbar
        cbar = plt.colorbar(cf, label=r'$\alpha_s$', orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(label=r'$\alpha_s$', fontsize=12)
        cbar.set_ticks([0.000, 0.006, 0.012])
        cbar.ax.set_position([0.75, 0.7, 0.2, 0.02])
        
        # Plot vectors with proper scaling
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        q = plt.quiver(
            xi[::skip_y, ::skip_x], yi[::skip_y, ::skip_x],
            ux_normalized[::skip_y, ::skip_x],
            uy_normalized[::skip_y, ::skip_x],
            speed[::skip_y, ::skip_x],
            scale=10/scale,  # Adjusted scaling (higher scale = shorter arrows)
            scale_units='inches',
            angles='uv',
            width=0.0002 * arrow_scale,
            headwidth=1 * arrow_scale,
            headlength=1 * arrow_scale,
            headaxislength=1 * arrow_scale,
            norm=norm,
            linewidth=1.0,
            cmap='cool',
            minlength=0.001  # Minimum length for arrows to appear
        )
        #plt.colorbar(q, label='Velocity Magnitude')
         # 添加速度颜色图例（quiver专用colorbar）
        cbar_speed = plt.colorbar(q, label='Velocity Magnitude [m/s]',orientation='horizontal', norm=norm)
        cbar_speed.ax.tick_params(labelsize=10)
        cbar_speed.set_label('Velocity Magnitude [m/s]', fontsize=12)
        cbar_speed.set_ticks([0, vmax/2, vmax])
        cbar_speed.ax.set_position([0.75, 0.8, 0.2, 0.02])  # [left, bottom, width, height]
        



        # 3. Overlay alpha contour
        c=plt.contour(xi, yi, alpha_i, **self.ALPHA_CONTOUR_PARAMS)
        

        # 4. Add reference lines/labels
        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color=self.colorset, linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.005, y_text, f'{label}', fontsize=20, zorder=3, color=self.colorset)



            # === 新增：绘制积分H连线 ===
        if h_points and len(h_points) > 0:
            h_x, h_y = zip(*h_points)
            
            # 按 x 坐标排序以确保连线顺序正确
            sorted_indices = np.argsort(h_x)
            h_x_sorted = np.array(h_x)[sorted_indices]
            h_y_sorted = np.array(h_y)[sorted_indices]
            
            # 绘制虚线
            plt.plot(
                h_x_sorted, h_y_sorted,
                color='red',               # 蓝色虚线（可自定义）
                linestyle='--',
                linewidth=2,
                label='Velocity Zero Crossing',
                zorder=4                    # 确保在箭头上方
            ) 



        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(*self.X_LIM if not normalize_x else (0,self.X_LIM[1]-head_x)/H0)
        plt.ylim(0,0.3)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()



    def measure_vortex_dimensions(self, xi, yi, u_rot, v_rot, alpha_i, min_vorticity=0.1):
        """
        量化每个涡旋的长宽尺寸（物理尺度）
        
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
        # 预处理：计算涡量场筛选候选区域
        dy = yi[1,0] - yi[0,0]  # y方向网格间距
        dx = xi[0,1] - xi[0,0]  # x方向网格间距
        
        # 计算速度梯度 (中心差分)
        duy = np.gradient(u_rot, dy, axis=0)  # ∂u'/∂y
        dvx = np.gradient(v_rot, dx, axis=1)          # ∂v/∂x
        
        # z方向涡量
        vorticity = dvx - duy
        mask = (np.abs(vorticity) > min_vorticity) & (alpha_i > self.alpha_threshold)
        
        # 连通区域标记（识别离散涡旋）
        from scipy.ndimage import label
        labeled, n_vortices = label(mask)
        
        vortices = []
        for i in range(1, n_vortices + 1):
            vortex_mask = (labeled == i)
            
            # 提取当前涡旋区域内的速度矢量
            points = np.column_stack([xi[vortex_mask], yi[vortex_mask]])
            velocities = np.column_stack([u_rot[vortex_mask], v_rot[vortex_mask]])
            
            # PCA分析
            pca = PCA(n_components=2)
            pca.fit(velocities - velocities.mean(axis=0))  # 去除平移运动
            
            # 计算几何参数
            center = np.mean(points, axis=0)  # 涡心坐标
            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_
            
            # 长宽尺寸正比于特征值平方根（能量权重）
            length = 2 * np.sqrt(eigenvalues[0])  # 长轴直径
            width = 2 * np.sqrt(eigenvalues[1])   # 短轴直径
            angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])  # 主轴角度
            
            vortices.append({
                'center': center,
                'length': length,
                'width': width,
                'angle': angle,
                'area': np.pi * (length/2) * (width/2)  # 椭圆面积近似
            })
        
        return vortices
    
    def plot_vortex_boundaries(self, xi, yi, vortices, u_rot, v_rot,time_v):
        """在速度场上叠加显示PCA测量的涡旋椭圆"""
        plt.figure(figsize=self.FIG_SIZE)
        
        # 绘制背景速度场（此处示例用streamplot）
        plt.streamplot(xi, yi, u_rot, v_rot, 
                    color='k', density=2, linewidth=0.5)
        
        # 绘制每个涡旋的椭圆
        from matplotlib.patches import Ellipse
        for vortex in vortices:
            ellipse = Ellipse(
                xy=vortex['center'],
                width=vortex['width'],
                height=vortex['length'],
                angle=np.degrees(vortex['angle']),
                edgecolor='r',
                facecolor='none',
                lw=2,
                linestyle='--'
            )
            plt.gca().add_patch(ellipse)
            
            # 标记中心点
            plt.scatter(*vortex['center'], c='r', s=50, marker='x')
        
        plt.title(f'Vortex Boundaries (PCA) at t={time_v}s')
        plt.xlim(self.X_LIM)
        plt.savefig(os.path.join(self.output_dir, f'vortex_pca_dimensions_t{time_v}.png'))
        plt.close()




    def process_time_step(self, time_v):
        """Process data for a single time step"""
        # Read field data
        Ua_A = fluidfoam.readvector(self.sol, str(time_v), "U.a")
        alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
        beta = fluidfoam.readscalar(self.sol, str(time_v), "alpha.b")
        gradU = fluidfoam.readtensor(self.sol, str(time_v), "grad(U.a)")
        vorticity = fluidfoam.readvector(self.sol, str(time_v), "vorticity")
        gradbeta = fluidfoam.readvector(self.sol, str(time_v), "grad(alpha.b)")
        gradvorticity = fluidfoam.readtensor(self.sol, str(time_v), "grad(vorticity)")

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
        X, Y, Z = fluidfoam.readmesh(self.sol)
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
        yi = np.linspace(Y.min(), Y.max(), 100)
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
                'Q_signed': self.signed_smooth(
                    self.calculate_q_criterion(gradU_x, gradU_y, gradV_x, gradV_y), 
                    omega_z
                ),
                
            }
        )

        # Define positions for vertical lines
        positions = {
            '$0.1H_0$': head_x - 0.1*self.Height,
            '$0.25H_0$': head_x - 0.25*self.Height,
            '$0.5H_0$': head_x - 0.5*self.Height,
            '$H_0$': head_x - self.Height
        }
        y_text = 0.32


                # 在process_time_step方法中（计算PCA前）
        v_rot = interpolated['uyi']  # 旋转速度y分量
        u_rot = interpolated['U_perturb']  # 旋转速度x分量

        # 测量所有涡旋尺寸
        vortex_properties = self.measure_vortex_dimensions(
            xi, yi,u_rot, v_rot, 
            interpolated['alpha_i'],
            min_vorticity=0.5  # 根据实际情况调整阈值
        )

        # 打印结果
        for i, vortex in enumerate(vortex_properties):
            print(f"涡旋 #{i}:")
            print(f"  中心位置: ({vortex['center'][0]:.3f}, {vortex['center'][1]:.3f}) m")
            print(f"  长轴长度: {vortex['length']:.3f} m")
            print(f"  短轴长度: {vortex['width']:.3f} m")
            print(f"  主轴角度: {np.degrees(vortex['angle']):.1f}°\n")

        # 可视化标记涡旋边界（可选）
        self.plot_vortex_boundaries(xi, yi, vortex_properties, u_rot, v_rot,time_v)

   
                #Add this after other plot calls in process_time_step
                # 从 results DataFrame 提取 H 高度的坐标点
        # h_points = [(row['x'], row['H_depth']) for _, row in df.iterrows()]
    #     self.plot_velocity_vectors(
    #         xi, yi, interpolated['uxi'], interpolated['uyi'], interpolated['alpha_i'],
    #         time_v, positions, y_text,
    #         title=f'Velocity Field (t={time_v}s)',
    #         filename=f'velocity_t{time_v}.png',
    #         skip_x=6,          # Adjust density (higher = sparser)
    #         skip_y=3,
    #         scale = 3,
    #         arrow_scale=1.0,
    #         alpha_opacity=0.7,
    #         alpha_cmap='gray_r',
    #         velocity_zero_points=velocity_zero_points,  # 新增参数
    #         h_points = h_points,  # H高度点
    #         normalize_x=False,  # 是否进行无量纲化
    #         head_x=head_x,    # 头部位置
    #         H0=self.Height,   # 初始高度
    #   # Or use speed for colormap (see below)
    #     )

        self.plot_velocity_vectors2(
            xi, yi, interpolated['uxi'], interpolated['uyi'], interpolated['alpha_i'],
            time_v, positions, y_text,
            title=f'Velocity Field (t={time_v}s)',
            filename=f'velocity_t{time_v}.png',
            skip_x=6,          # Adjust density (higher = sparser)
            skip_y=3,
            scale = 3,
            arrow_scale=1.0,
            alpha_opacity=0.7,
            alpha_cmap='gray_r',
            velocity_zero_points=velocity_zero_points,  # 新增参数
            h_points = h_points,  # H高度点
            normalize_x=False,  # 是否进行无量纲化
            head_x=head_x,    # 头部位置
            H0=self.Height,   # 初始高度
      # Or use speed for colormap (see below)
        )
        
    #     self.plot_velocity_vectors3(
    #         xi, yi, interpolated['U_perturb'], interpolated['uyi'], interpolated['alpha_i'],
    #         time_v, positions, y_text,
    #         title=f'Rotation velocity Field (t={time_v}s)',
    #         filename=f'rotation_velocity_t{time_v}.png',
    #         skip_x=6,          # Adjust density (higher = sparser)
    #         skip_y=3,
    #         scale = 5,
    #         arrow_scale=5,
    #         alpha_opacity=0.7,
    #         alpha_cmap='gray_r',
    #         velocity_zero_points=velocity_zero_points,  # 新增参数
    #         h_points = h_points,  # H高度点
    #         normalize_x=False,  # 是否进行无量纲化
    #         head_x=head_x,    # 头部位置
    #         H0=self.Height,   # 初始高度
    #         Hi=H_depth
            
    #   # Or use speed for colormap (see below)
    #     )




        # Generate plots
        self.plot_streamlines(
            xi, yi, interpolated['U_perturb'], interpolated['uyi'], 
            interpolated['U_perturb'], interpolated['alpha_i'], time_v, 
            positions, y_text, f'Rotation Velocity Streamlines (t={time_v})',
            f'Rotation_streamlines_t{time_v}s.png', 
            "$\hat{U}_s$ [m/s]", -0.3, 0.07
        )

        # self.plot_streamlines(
        #     xi, yi, interpolated['U_perturb'], interpolated['uyi'], 
        #     interpolated['U_perturb_alpha'], interpolated['alpha_i'], time_v, 
        #     positions, y_text, f'Perturbation Velocity Streamlines (t={time_v})',
        #     f'perturbation_streamlines_ALPHA_t{time_v}.png', 
        #     "Velocity Perturbation $U_aALPHA\'$ [m/s]", -0.2, 0.05
        # )

        self.plot_streamlines2(
            xi, yi, interpolated['uxi'], interpolated['uyi'], 
            interpolated['uxi'], interpolated['alpha_i'], time_v, 
            positions, y_text, f'Original Velocity Streamlines (t={time_v})',
            f'origin_streamlines_t{time_v}.png', 
            "Velocity $U_s$ [m/s]", -0.1, 0.1
        )

        # self.plot_contour(
        #     xi, yi, interpolated['Uper_densitygradient'], interpolated['alpha_i'], 
        #     time_v, positions, y_text, f'DensityGradientPerturb (t={time_v})',
        #     f'DensityGradientPerturb_t{time_v}.png', 'DensityGradientPerturb',
        #     levels=np.linspace(-0.5, 0.5, 21)
        # )

        # self.plot_contour(
        #     xi, yi, interpolated['Umean_densitygradient'], interpolated['alpha_i'], 
        #     time_v, positions, y_text, f'DensityGradientMean (t={time_v})',
        #     f'DensityGradientMean_t{time_v}.png', 'DensityGradientMean',
        #     levels=np.linspace(-0.5, 0.5, 21)
        # )

        self.plot_contour(
            xi, yi, interpolated['omega_z'], interpolated['alpha_i'], 
            time_v, positions, y_text, fr'$\alpha_s$ (t={time_v})',
            f'contour_t{time_v}.png', r'$\alpha_s$ ',
            # levels=np.linspace(-10.5, 10.5, 81),
            # cmap='bwr'
        )

        # self.plot_contour(
        #     xi, yi, interpolated['Q_signed'], interpolated['alpha_i'], 
        #     time_v, positions, y_text, 
        #     f'Vortex identification via Q_criterion (t={time_v})',
        #     f'Q_criterion_t{time_v}_signed_omega.png', 
        #     'Q-Criterion',
        #     levels=np.linspace(-5, 5, 41),
        #     cmap='bwr'
        # )

        # self.plot_contour(
        #     xi, yi, interpolated['Umean_advection'], interpolated['alpha_i'], 
        #     time_v, positions, y_text, 
        #     f'Umean_advection (t={time_v})',
        #     f'Umean_advection_t{time_v}.png', 
        #     'Umean_advection',
        #     levels=np.linspace(-10, 10, 21),
        #     cmap='bwr'
        # )


        # self.plot_contour(
        #     xi, yi, interpolated['Uper_advection'], interpolated['alpha_i'], 
        #     time_v, positions, y_text, 
        #     f'Uper_advection (t={time_v})',
        #     f'Uper_advection_t{time_v}.png', 
        #     'Uper_advection',
        #     levels=np.linspace(-10, 10, 21),
        #     cmap='bwr'
        # )


        # self.plot_contour(
        #     xi, yi, interpolated['uxi'], interpolated['alpha_i'], 
        #     time_v, positions, y_text, 
        #     f'Uori_advection (t={time_v})',
        #     f'Uori_advection_t{time_v}.png', 
        #     'Uori_advection',
        #     levels=np.linspace(-10, 10, 21),
        #     cmap='bwr'
        # )

        self.plot_vorticity_contour(
            xi, yi, 
            interpolated['U_perturb'],         # 原始x方向速度
            interpolated['uyi'],         # 原始y方向速度
            interpolated['alpha_i'],     # 颗粒浓度
            time_v,
            positions, 
            y_text,
            title=f'Vorticity Field ($\hat u$ , t={time_v}s)',
            filename=f'vorticity_rotation_t{time_v}.png',
            levels=np.linspace(-5, 5, 41),  # 自定义色阶范围
            cmap='bwr'                       # 红蓝配色
        )

        self.plot_vorticity_contour(
            xi, yi, 
            interpolated['uxi'],         # 原始x方向速度
            interpolated['uyi'],         # 原始y方向速度
            interpolated['alpha_i'],     # 颗粒浓度
            time_v,
            positions, 
            y_text,
            title=f'Vorticity Field (original u, t={time_v}s)',
            filename=f'vorticity_t{time_v}.png',
            levels=np.linspace(-5, 5, 41),  # 自定义色阶范围
            cmap='bwr'                       # 红蓝配色
        )


    def run_analysis(self):
        """Main method to run the analysis for all time steps"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        for time_v in self.times:
            print(f"Processing time step: {time_v}")
            self.process_time_step(time_v)
        
        print(f"All results saved to: {self.output_dir}")


if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()