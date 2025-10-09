import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm

class TurbidityCurrentAnalyzer:
    def __init__(self):
        # Configuration parameters
        self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
        # self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090912_1"
        self.output_dir = "/home/amber/postpro/u_umean_tc"
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.times = [10]
        self.FIG_SIZE = (40, 8)
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
        
        # Initialize global plot settings
        plt.rcParams.update({
            'font.size': 28,
            'axes.titlesize': 28,
            'axes.labelsize': 24,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 24
        })


    def interpolate_fields(self, X, Y, xi, yi, fields):
        """Interpolate fields to regular grid"""
        return {name: griddata((X, Y), field, (xi, yi), method='linear') 
                for name, field in fields.items()}

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
        
        # Vectorized integration

        integralU = np.trapz(ua_x[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU2 = np.trapz(ua_x[:max_ya_crossing_index]**2, ya[:max_ya_crossing_index])
        integral2 = np.trapz((ua_x[:max_ya_crossing_index] * alpha_vals[:max_ya_crossing_index])**2, ya[:max_ya_crossing_index])
        
        # Calculate derived quantities
        U = integralU2 / integralU if integralU != 0 else 0
        H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0
        print('max_ya_crossing_index:', ya[max_ya_crossing_index],'at max_ya_crossing_index:',max_ya_crossing_index)
        print('uavalue:', ua_x[max_ya_crossing_index])
        
        return U, H_depth, ya[max_ya_crossing_index]

    






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

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(*self.X_LIM if not normalize_x else (0,self.X_LIM[1]-head_x)/H0)
        plt.ylim(0,0.3)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()




    def process_time_step(self, time_v):
        """Process data for a single time step"""
        # Read field data
        Ua_A = fluidfoam.readvector(self.sol, str(time_v), "U.a")
        alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")


        # Extract components



        velocity_zero_points = []

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
            mask = (X == xx)  & (Y >= 0) #& (alpha_A >= 1e-5)
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
            U, H_depth, y_crossing = self.integrate_quantities(ya, ua_x, alpha)
            # 存储速度零点的 (x, y)
            velocity_zero_points.append((xx, y_crossing))
            results.append({
                "Time": time_v,
                "x": xx,
                # "ya": ya,
                "U": U,
                
                "y_crossing": y_crossing,
                
    
               
            })
        # 在 process_time_step 方法中添加：
        xtarget = 1.26
        mask = np.isclose(X, xtarget, atol=1e-4)  # 容差匹配
        if np.any(mask):
            ya = Y[mask]
            ua_x = Ua_A[0][mask]
            alpha = alpha_A[mask]
            
            # 计算 y_crossing（复用你的 integrate_quantities 方法）
            _, _, y_crossing = self.integrate_quantities(ya, ua_x, alpha)
            print(f"At x={xtarget}m, y_crossing = {y_crossing:.4f}m")
            

        else:
            print(f"Warning: No data found at x={xtarget}m")




        # Save results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, f"integration_results_t{time_v}.csv"), index=False)
        print(f"Results saved for t={time_v} (head_x={head_x:.2f}m)")

                # 方法1：直接使用原始场数据 Ua_A[0]
        # mask = (X == 1.26)  # 可选添加 y_min 限制
        # y_values = Y[mask]
        # u_values = Ua_A[0][mask]  # 直接从完整场提取
        
        # plt.figure(figsize=(8, 6))
        # plt.plot(u_values, y_values, 'b-o', linewidth=2)
        # plt.xlabel('Velocity u [m/s]')
        # plt.ylabel('Height y [m]')
        # plt.title(f'Velocity Profile at x=1.26m (t={time_v}s)')
        # plt.grid(True)
        # plt.title('titlee')
        # plt.savefig(os.path.join(self.output_dir, 'title'), dpi=300, bbox_inches='tight')
        # #plt.close()

        # # 保存到CSV
        # curve_data = pd.DataFrame({'y': y_values, 'u': u_values})
        # curve_data.to_csv(
        #     os.path.join(self.output_dir, f"u_y_curve_x_t{time_v}.csv"),
        #     index=False
        # )


        


        # Interpolate fields
        xi = np.linspace(X.min(), X.max(), 750)#1000
        yi = np.linspace(Y.min(), Y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        interpolated = self.interpolate_fields(
            X, Y, xi, yi, {
                
                'uxi': Ua_A[0],
                'uyi': Ua_A[1],
                'alpha_i': alpha_A,
 
                
            }
        )

        # Define positions for vertical lines
        positions = {
            '0.1H': head_x - 0.1*self.Height,
            '0.25H': head_x - 0.25*self.Height,
            '0.5H': head_x - 0.5*self.Height,
            'H': head_x - self.Height
        }
        y_text = 0.32
   
                #Add this after other plot calls in process_time_step
                # 从 results DataFrame 提取 H 高度的坐标点
        
        self.plot_velocity_vectors(
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
             # H高度点
            normalize_x=False,  # 是否进行无量纲化
            head_x=head_x,    # 头部位置
            H0=self.Height,   # 初始高度
      # Or use speed for colormap (see below)
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