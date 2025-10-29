import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
# sklearn.decomposition.PCA 不再需要，因为我们有ManualPCA
from scipy.ndimage import label, binary_closing
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


# ==============================================================================
#  工具类：手动实现的PCA (保持不变)
# ==============================================================================
class ManualPCA:
    """手动实现的PCA分析，用于涡旋几何特征量化"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X):
        if X is None or X.shape[0] < 3 or not np.all(np.isfinite(X)):
            raise ValueError(f"输入到PCA的数据无效 (点数: {X.shape[0]}, 包含NaN/inf: {not np.all(np.isfinite(X))})。")
        self.mean_ = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        sort_indices = np.argsort(eigvals)[::-1]
        self.components_ = eigvecs[:, sort_indices]
        self.explained_variance_ = eigvals[sort_indices]
        return self

    def get_vortex_properties(self):
        if self.components_ is None: raise ValueError("请先调用fit()方法")
        length = 4 * np.sqrt(self.explained_variance_[0])
        width = 4 * np.sqrt(self.explained_variance_[1])
        major_axis = self.components_[:, 0]
        if major_axis[0] < 0: major_axis *= -1
        angle = np.arctan2(major_axis[1], major_axis[0])
        return {'center': self.mean_, 'length': length, 'width': width, 'angle': angle,
                'major_axis': major_axis, 'minor_axis': self.components_[:, 1]}


# ==============================================================================
#  主分析类：瞬时场分析 (有少量但关键的修改)
# ==============================================================================
class TurbidityCurrentAnalyzer:
    def __init__(self):
        self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
        self.output_dir = "/home/amber/postpro/u_umean_tc"
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.times = [7, 10, 12]
        self.FIG_SIZE = (40, 6)
        self.X_LIM = (0.0, 1.6); self.Y_LIM = (0.0, 0.3)
        self.Height = 0.3
        self.colorset = 'fuchsia'
        
        self.ALPHA_CONTOUR_PARAMS = {'levels': [1e-5], 'colors': 'black', 'linewidths': 2, 'linestyles': 'dashed', 'zorder': 3}
        
        plt.rcParams.update({'font.size': 28, 'axes.titlesize': 28, 'axes.labelsize': 24,
                             'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 24})

    def calculate_q_criterion(self, ux, uy, dx, dy):
        dUx = np.gradient(ux, dx, axis=1)
        dUy = np.gradient(ux, dy, axis=0)
        dVx = np.gradient(uy, dx, axis=1)
        dVy = np.gradient(uy, dy, axis=0)
        
        S_xx, S_yy = dUx, dVy
        S_xy = S_yx = 0.5 * (dUy + dVx)
        Omega_xy = 0.5 * (dVx - dUy)
        Omega_yx = -Omega_xy
        
        S_norm = S_xx**2 + 2 * S_xy**2 + S_yy**2
        Omega_norm = 2 * Omega_xy**2
        Q = 0.5 * (Omega_norm - S_norm)
        return np.nan_to_num(Q, nan=0, posinf=0, neginf=0)
    
    # --- 新增的、为POD服务的方法 (保持不变) ---
    def get_perturbation_fields_for_time(self, time_v, xi, yi):
        print(f"  (POD Prep) Processing t={time_v} to get perturbation fields...")
        try:
            Ua_A = fluidfoam.readvector(self.sol, str(time_v), "U.a")
            alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
            X, Y, _ = fluidfoam.readmesh(self.sol)

            head_x = None
            for x in reversed(np.unique(X)):
                mask = (X == x) & (Y >= self.y_min) & (alpha_A > self.alpha_threshold)
                if np.any(mask):
                    head_x = x
                    break
            if head_x is None: return None, None

            results = []
            x_coords = np.unique(X[(X <= head_x) & (X >= 0)])
            for xx in x_coords:
                mask = (X == xx) & (Y >= 0) & (alpha_A > 1e-5)
                if not np.any(mask): continue
                ya, ua = Y[mask], Ua_A[0][mask]
                integralU = np.trapz(ua, ya)
                integralU2 = np.trapz(ua**2, ya)
                U = integralU2 / integralU if integralU != 0 else 0
                results.append({"x": xx, "U": U})
            
            if not results: return None, None
            df = pd.DataFrame(results)
            x_U_mapping = dict(zip(df['x'], df['U']))

            U_perturb_raw = Ua_A[0].copy()
            for x_val in np.unique(X):
                if x_val in x_U_mapping:
                    mask = (X == x_val)
                    U_perturb_raw[mask] = Ua_A[0][mask] - x_U_mapping[x_val]

            u_perturb_interp = griddata((X, Y), U_perturb_raw, (xi, yi), method='linear', fill_value=0)
            v_interp = griddata((X, Y), Ua_A[1], (xi, yi), method='linear', fill_value=0)
            
            return u_perturb_interp, v_interp
        except Exception as e:
            print(f"  Error processing t={time_v}: {e}")
            return None, None

    # --- 对您原始的 measure_vortex_dimensions 进行了关键修改 ---
    def measure_vortex_dimensions(self, xi, yi, ux, uy, alpha_i, Q_field):
        print("Measuring vortex dimensions...")
        q_max = np.nanmax(Q_field)
        if q_max <= 0: return []
            
        q_threshold_high = q_max * 1e-2  # 稍微放宽种子阈值
        q_threshold_low = q_max * 1e-3   # 稍微放宽边界阈值

        initial_mask = (Q_field > q_threshold_high) & (alpha_i > self.alpha_threshold)
        structure = np.ones((5, 5), dtype=bool)
        closed_mask = binary_closing(initial_mask, structure=structure)
        final_mask = closed_mask & (Q_field > q_threshold_low)
        
        labeled, n_vortices = label(final_mask)
        print(f"  Identified {n_vortices} vortex structures after morphological operations.")
    
        vortices = []
        for i in range(1, n_vortices + 1):
            vortex_mask = (labeled == i)
            if np.sum(vortex_mask) < 10: continue # 增加一个最小点数过滤

            points = np.column_stack([xi[vortex_mask], yi[vortex_mask]])
            velocities = np.column_stack([ux[vortex_mask], uy[vortex_mask]])
            
            try:
                pca_geo = ManualPCA().fit(points)
                geo_props = pca_geo.get_vortex_properties()
                
                pca_kin = ManualPCA().fit(velocities - velocities.mean(axis=0))
                kin_props = pca_kin.get_vortex_properties()
            except ValueError as e:
                print(f"  Skipping vortex #{i} due to PCA error: {e}")
                continue

            vortices.append({
                'center': geo_props['center'], 'length': geo_props['length'], 'width': geo_props['width'],
                'geo_angle': geo_props['angle'], 'geo_major_axis': geo_props['major_axis'],
                'kin_angle': kin_props['angle'], 'kin_major_axis': kin_props['major_axis'],
                # (为简洁，省略kin_center, kin_length等)
            })
        return vortices

    def plot_vortex_boundaries(self, xi, yi, vortices, ux, uy, time_v, alpha_i):
        plt.figure(figsize=self.FIG_SIZE)
        mask = alpha_i > 1e-5
        ux_masked = np.where(mask, ux, 0)
        uy_masked = np.where(mask, uy, 0)
        plt.streamplot(xi, yi, ux_masked, uy_masked, color='#0343df', linewidth=1, density=5, zorder=2)
        
        for i, vortex in enumerate(vortices):
            ellipse = Ellipse(xy=vortex['center'], width=vortex['length'], height=vortex['width'],
                              angle=np.degrees(vortex['geo_angle']), edgecolor='r', facecolor='none',
                              linestyle='--', linewidth=2, zorder=3)
            plt.gca().add_patch(ellipse)
            
            p1 = vortex['center'] - 0.5 * vortex['length'] * vortex['geo_major_axis']
            p2 = vortex['center'] + 0.5 * vortex['length'] * vortex['geo_major_axis']
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1.5, zorder=4, label='Geometric Axis' if i==0 else "")
            
            # 确保 'kin_major_axis' 存在
            if 'kin_major_axis' in vortex and vortex['kin_major_axis'] is not None:
                p3 = vortex['center'] - 0.5 * vortex['length'] * vortex['kin_major_axis']
                p4 = vortex['center'] + 0.5 * vortex['length'] * vortex['kin_major_axis']
                plt.plot([p3[0], p4[0]], [p4[1], p4[1]], 'magenta', linewidth=1.5, zorder=4, label='Kinematic Axis' if i==0 else "")

            plt.scatter(*vortex['center'], c='r', s=50, marker='x', zorder=5)
            plt.text(vortex['center'][0], vortex['center'][1], str(i), color='k', fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=6)
        
        plt.title(f'Vortex Boundaries (PCA Analysis) at t={time_v}s')
        plt.xlim(self.X_LIM); plt.ylim(self.Y_LIM)
        legend_elements = [Line2D([0], [0], color='r', linestyle='--', lw=2, label='Geometric Shape'),
                           Line2D([0], [0], color='magenta', lw=1.5, label='Kinematic Axis'),
                           Line2D([0], [0], marker='x', color='r', lw=0, label='Vortex Center')]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.savefig(os.path.join(self.output_dir, f'vortex_pca_dimensions_t{time_v}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # --- 对您的 process_time_step 进行了关键修改 ---
    def process_time_step(self, time_v):
        print(f"Processing snapshot at t={time_v}...")
        X, Y, _ = fluidfoam.readmesh(self.sol)
        xi = np.linspace(X.min(), X.max(), 750)
        yi = np.linspace(Y.min(), Y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)

        # 复用为POD写的方法来获取扰动场
        u_perturb, uyi = self.get_perturbation_fields_for_time(time_v, xi, yi)
        if u_perturb is None:
            print(f"  Could not generate perturbation field for t={time_v}. Skipping.")
            return
            
        # 获取 alpha 场用于遮罩
        alpha_i = griddata((X, Y), fluidfoam.readscalar(self.sol, str(time_v), "alpha.a"), (xi, yi), method='linear', fill_value=0)

        # --- 关键补充：计算Q准则场 ---
        dx = xi[0, 1] - xi[0, 0]
        dy = yi[1, 0] - yi[0, 0]
        Q_field = self.calculate_q_criterion(u_perturb, uyi, dx, dy)

        # 测量所有涡旋尺寸 (现在输入是正确的)
        vortex_properties = self.measure_vortex_dimensions(
            xi, yi, u_perturb, uyi, alpha_i, Q_field
        )

        # 打印结果
        for i, vortex in enumerate(vortex_properties):
            print(f"  Vortex #{i}: Center=({vortex['center'][0]:.3f}, {vortex['center'][1]:.3f}), "
                  f"Length={vortex['length']:.3f}, Width={vortex['width']:.3f}, "
                  f"GeoAngle={np.degrees(vortex['geo_angle']):.1f}°")

        # 可视化标记涡旋边界
        self.plot_vortex_boundaries(xi, yi, vortex_properties, u_perturb, uyi, time_v, alpha_i)
        
        # (您的其他绘图函数，如plot_streamlines等，可以在这里调用)

    def run_analysis(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for time_v in self.times:
            print(f"\n--- Processing Instantaneous Snapshot: t={time_v} ---")
            try:
                self.process_time_step(time_v)
            except Exception as e:
                print(f"    FATAL ERROR processing t={time_v}: {e}")
        print(f"\nInstantaneous analysis complete. Results saved to: {self.output_dir}")


# ==============================================================================
#  新类：POD 时序分析 (经过大幅修正和简化)
# ==============================================================================
class TurbidityCurrentPODAnalyzer:
    def __init__(self, base_analyzer):
        self.base_analyzer = base_analyzer
        self.sol = base_analyzer.sol
        self.output_dir = os.path.join(base_analyzer.output_dir, "POD_analysis_perturbation")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"POD analysis results will be saved to: {self.output_dir}")
        
        self.time_start = 5
        self.time_end = 15
        self.time_step = 0.5
        
        self.n_x, self.n_y = 750, 200
        self.xi, self.yi = self._initialize_grid()
        
    def _initialize_grid(self):
        X, Y, _ = fluidfoam.readmesh(self.sol)
        xi_coords = np.linspace(X.min(), X.max(), self.n_x)
        yi_coords = np.linspace(Y.min(), Y.max(), self.n_y)
        return np.meshgrid(xi_coords, yi_coords)

    def _get_time_steps(self):
        try:
            all_times = [float(d) for d in os.listdir(self.sol) if d.replace('.', '', 1).isdigit()]
            valid_times = [t for t in all_times if self.time_start <= t <= self.end_time]
            valid_times.sort()
            return [str(t) for t in valid_times if t % self.time_step == 0]
        except: # 简化版
            return [str(t) for t in np.arange(self.time_start, self.time_end + self.time_step, self.time_step)]

    def run_pod_analysis(self):
        time_steps = self._get_time_steps()
        if not time_steps:
            print("Error: No valid time steps found.")
            return
            
        print(f"Found {len(time_steps)} time steps for POD analysis.")

        print("\nStep 1 & 2: Collecting perturbation snapshots and calculating temporal mean...")
        
        num_grid_points = self.n_x * self.n_y
        mean_field_sum = np.zeros((2, self.n_y, self.n_x))
        all_snapshots = []
        valid_time_steps = []
        
        for t in time_steps:
            u_perturb, v = self.base_analyzer.get_perturbation_fields_for_time(t, self.xi, self.yi)
            if u_perturb is None:
                print(f"  Skipping t={t}, could not compute.")
                continue
            
            valid_time_steps.append(t)
            snapshot = np.stack([u_perturb, v])
            mean_field_sum += snapshot
            all_snapshots.append(snapshot)

        if not all_snapshots:
            print("Error: No valid snapshots could be processed."); return

        mean_field = mean_field_sum / len(all_snapshots)
        
        print("\nStep 3: Constructing fluctuation data matrix...")
        M, T = 2 * num_grid_points, len(all_snapshots)
        A = np.zeros((M, T))
        
        for i, snapshot in enumerate(all_snapshots):
            fluctuation = (snapshot - mean_field).flatten()
            A[:, i] = fluctuation
            
        print(f"Data matrix A created with shape: {A.shape}")
        
        print("\nStep 4: Performing SVD...")
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        print("SVD complete.")
        
        print("\nStep 5: Post-processing and visualizing results...")
        # self.plot_energy_spectrum(S)
        self.plot_spatial_mode(mean_field, -1) 
        for i in range(min(4, len(S))):
            self.plot_spatial_mode(U, i, num_grid_points)
        # self.plot_temporal_coefficients(Vt, S, valid_time_steps, 4)

    # def plot_energy_spectrum(self, S):
               
    #     """绘制能量谱图"""
    #     plt.figure()
    #     plt.plot(S, 'o-')
    #     plt.title(f'POD Energy Spectrum - {S} component')
    #     plt.xlabel('Mode number')
    #     plt.ylabel('Energy ratio')
    #     plt.savefig(os.path.join(self.output_dir, f'pod_energy_spectrum_{S}.png'))
    #     plt.close()
    #     pass

    def plot_spatial_mode(self, data, mode_index, num_grid_points=None):
        """可视化空间模态 (修正后)"""
        plt.figure(figsize=(20, 5))
        
        # --- 这里的逻辑是正确的，根据mode_index判断画什么 ---
        if mode_index == -1:
            u_mode, v_mode = data[0], data[1]
            title = 'Mean Field (Mode 0)'
            filename = 'pod_mode_0_mean_field.png'
        else:
            mode_vector = data[:, mode_index]
            u_mode = mode_vector[:num_grid_points].reshape((self.n_y, self.n_x))
            v_mode = mode_vector[num_grid_points:].reshape((self.n_y, self.n_x))
            title = f'POD Spatial Mode #{mode_index + 1}'
            filename = f'pod_mode_{mode_index + 1}.png'
            
        magnitude = np.sqrt(u_mode**2 + v_mode**2)
        vmax = np.nanmax(magnitude)
        
        # 确保 vmax > 0 避免绘图错误
        if vmax > 1e-9: # 使用一个很小的值来避免浮点误差
            plt.contourf(self.xi, self.yi, magnitude, levels=100, cmap='viridis', vmin=0, vmax=vmax)
            plt.colorbar(label='Mode Magnitude')
            
            # 修正流线图的线宽计算
            speed = magnitude # speed 和 magnitude 是一样的
            # 创建一个归一化的线宽数组，避免过粗或过细
            lw = 2.5 * speed / vmax
            lw[speed < 0.1 * vmax] = 0.5 # 对速度小的区域使用较细的线

            plt.streamplot(self.xi, self.yi, u_mode, v_mode, color='white', density=1.5, linewidth=lw)
        else:
            # 如果整个场都是0，就只画一个空白图
            plt.text(0.5, 0.5, 'Zero Field', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.title(title)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    # def plot_temporal_coefficients(self, Vt, S, time_steps, num_modes_to_plot):
    #     # ... (此方法与上一版完全相同，复制到此处)
    #     pass


# ==============================================================================
#  主程序入口 (修正后)
# ==============================================================================
if __name__ == "__main__":
    
    # --- 步骤 1: 创建一个基础分析器实例 ---
    base_analyzer = TurbidityCurrentAnalyzer()
    
    # --- 步骤 2: (推荐) 运行瞬时场分析 ---
    # 这将为 self.times 中定义的几个时刻生成详细的涡椭圆图
    print("--- STARTING INSTANTANEOUS VORTEX ANALYSIS ---")
    base_analyzer.run_analysis()
    print("\nInstantaneous analysis complete.\n" + "="*50 + "\n")
    
    # --- 步骤 3: (可选) 运行POD时序分析 ---
    # 这将分析一个时间段内的主要流动模式
    print("--- STARTING POD ANALYSIS ---")
    pod_analyzer = TurbidityCurrentPODAnalyzer(base_analyzer)
    pod_analyzer.run_pod_analysis()
    print("\nPOD analysis complete.")