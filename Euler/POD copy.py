import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from scipy.ndimage import label, binary_closing
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


# ==============================================================================
#  工具类：手动实现的PCA
# ==============================================================================
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
        # --- 防线：检查输入数据 ---
        if X is None or X.shape[0] < 3 or not np.all(np.isfinite(X)):
            raise ValueError("输入到PCA的数据无效 (点数少于3, 或包含 NaN/inf)。")

        self.mean_ = np.mean(X, axis=0)
        cov_matrix = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov_matrix)
        
        sort_indices = np.argsort(eigvals)[::-1]
        self.components_ = eigvecs[:, sort_indices]
        self.explained_variance_ = eigvals[sort_indices]
        
        return self

    def get_vortex_properties(self):
        """
        返回涡旋的几何或动力学特征
        """
        if self.components_ is None:
            raise ValueError("请先调用fit()方法")
            
        length = 4 * np.sqrt(self.explained_variance_[0])
        width = 4 * np.sqrt(self.explained_variance_[1])
        
        major_axis = self.components_[:, 0]
        # 确保主轴方向的唯一性（例如，让x分量始终为正）
        if major_axis[0] < 0:
            major_axis *= -1
        
        angle = np.arctan2(major_axis[1], major_axis[0])
        
        return {
            'center': self.mean_,
            'length': length,
            'width': width,
            'angle': angle,
            'major_axis': major_axis,
            'minor_axis': self.components_[:, 1]
        }


# ==============================================================================
#  主分析类：瞬时场分析
# ==============================================================================
class TurbidityCurrentAnalyzer:
    def __init__(self):
        # Configuration parameters
        self.sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
        self.output_dir = "/home/amber/postpro/u_umean_tc"
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.times = [7, 10, 12]
        self.FIG_SIZE = (40, 6)
        self.X_LIM = (0.0, 1.6)
        self.Y_LIM = (0.0, 0.3)
        self.Height = 0.3
        self.colorset = 'fuchsia'
        self.Q_threshold = 0.5  # 这是一个固定的阈值，可能需要调整
        
        # Visualization parameters
        self.ALPHA_CONTOUR_PARAMS = { 'levels': [1e-5], 'colors': 'black', 'linewidths': 2, 'linestyles': 'dashed', 'zorder': 3 }
        self.ALPHA_CONTOUR_PARAMS2 = { 'levels': [1e-3], 'colors': 'blueviolet', 'linewidths': 2, 'linestyles': 'dashed', 'zorder': 3 }
        
        plt.rcParams.update({
            'font.size': 28, 'axes.titlesize': 28, 'axes.labelsize': 24,
            'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 24
        })

    # --- 您所有的原始方法都保留在这里 ---
    def calculate_q_criterion(self, dUx, dUy, dVx, dVy):
        S_xx, S_yy = dUx, dVy
        S_xy = S_yx = 0.5 * (dUy + dVx)
        Omega_xy = 0.5 * (dVx - dUy)
        Omega_yx = -Omega_xy
        S_norm = S_xx**2 + 2 * S_xy**2 + S_yy**2
        Omega_norm = 2 * Omega_xy**2
        Q = 0.5 * (Omega_norm - S_norm)
        return np.nan_to_num(Q, nan=0, posinf=0, neginf=0)

    # --- 这是一个新增的、为POD服务的方法 ---
    def get_perturbation_fields_for_time(self, time_v, xi, yi):
        """
        一个专门为POD准备的方法。
        它为单个时间步计算并返回插值后的扰动速度场。
        """
        print(f"  (POD Prep) Processing t={time_v} to get perturbation fields...")
        try:
            Ua_A = fluidfoam.readvector(self.sol, str(time_v), "U.a")
            alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
            X, Y, _ = fluidfoam.readmesh(self.sol)

            head_x = None
            unique_x = np.unique(X)
            for x in reversed(unique_x):
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
                    U_mean = x_U_mapping[x_val]
                    U_perturb_raw[mask] = Ua_A[0][mask] - U_mean

            u_perturb_interp = griddata((X, Y), U_perturb_raw, (xi, yi), method='linear', fill_value=0)
            v_interp = griddata((X, Y), Ua_A[1], (xi, yi), method='linear', fill_value=0)
            
            return u_perturb_interp, v_interp
        except Exception as e:
            print(f"  Error processing t={time_v}: {e}")
            return None, None

    # --- 您的其他所有方法 (plot_*, measure_vortex_dimensions, process_time_step, etc.) 都保持不变 ---
    # ... (此处省略您文件中的所有其他方法，请将它们原封不动地复制到这里) ...
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



    def measure_vortex_dimensions(self, xi, yi, ux, uy, alpha_i, min_vorticity=8):
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
            
        q_threshold_high = q_max * 1e-3  # 用来找可靠的“种子”
        q_threshold_low = q_max * 1e-4    # 用来定义“焊接”的边界

    # ========== 2. 创建初始掩码 ==========
    # 这是我们严格识别出的、但可能破碎的涡核
        initial_mask = (Q > q_threshold_high) &  (alpha_i > self.alpha_threshold)

    # ========== 3. 使用“闭运算”进行焊接 ==========
    # 关键：我们现在有了焊接的边界！闭运算只会在 Q > q_threshold_low 的区域内生效。
    # (虽然binary_closing本身不直接使用边界，但它的效果等同于此)
    
    # 结构元素的大小决定了“焊条”的长度
    # 如果断裂带很宽，就需要更大的structure
        structure = np.ones((5, 5), dtype=bool) 

    # 对初始的、破碎的掩码进行闭运算
        closed_mask = binary_closing(initial_mask, structure=structure)

    # (可选，但推荐) 确保焊接后的区域仍然在宽容的边界内
        final_mask = closed_mask & (Q > q_threshold_low)
    
    # ========== 4. 在最终处理过的掩码上标记连通区域 ==========
    # 此时，原本破碎的区域应该已经被连接成了一个整体
        # final_mask_pos = final_mask & (vorticity > 0)
        # final_mask_neg = final_mask & (vorticity < 0)

        labeled, n_vortices = label(final_mask)
 
        
        
        
        # 连通区域标记（识别离散涡旋）



        # # ========== 2. 标记连通区域 ==========
        # # 检测 Q > Q_threshold 的连通区域
        # Q_mask = (Q > self.Q_threshold) & (alpha_i > self.alpha_threshold)
        # labeled, n_vortices = label(Q_mask)
        


        print(f"通过闭运算连接后，识别出 {n_vortices} 个涡结构。")
    
        # ========== 3. 逐个分析涡旋 ==========
        vortices = []
        for i in range(1, n_vortices + 1):
            vortex_mask = (labeled == i)
            points = np.column_stack([xi[vortex_mask], yi[vortex_mask]])
            velocities = np.column_stack([ux[vortex_mask], uy[vortex_mask]])

            # ---------- 关键校验：确保数据有效 ----------
            # if len(points) < 2:
            #     print(f"警告：涡旋 {i} 点数不足（{len(points)}），跳过")
            #     continue
            
            # if np.isnan(points).any() or np.isinf(points).any():
            #     print(f"警告：涡旋 {i} 含NaN/inf，跳过")
            #     continue

            # ---------- 几何PCA分析 ----------
            try:
                pca_geo = ManualPCA()
                pca_geo.fit(points)
                geo_props = pca_geo.get_vortex_properties()
            except np.linalg.LinAlgError as e:
                print(f"涡旋 {i} 几何PCA失败：{e}")
                continue

            # ---------- 动力学PCA分析 ----------
            try:
                pca_kin = ManualPCA()
                pca_kin.fit(velocities - velocities.mean(axis=0))
                kin_props = pca_kin.get_vortex_properties()
            except Exception as e:
                print(f"涡旋 {i} 动力学PCA失败：{e}")
                kin_props = {'angle': np.nan, 'major_axis': None}

            vortices.append({
                # 几何特征
                'center': geo_props['center'],
                'length': geo_props['length'],
                'width': geo_props['width'],
                'geo_angle': geo_props['angle'],
                'geo_major_axis': geo_props['major_axis'],
                
                # 动力学特征
                'kin_center': kin_props['center'],
                'kin_length': kin_props['length'],
                'kin_width': kin_props['width'],
                'kin_angle': kin_props.get('angle', np.nan),
                'kin_major_axis': kin_props.get('major_axis', None),
                
                # 其他属性
                'area': np.pi * (geo_props['length']/2) * (geo_props['width']/2),
                'max_vorticity': np.max(np.abs(Q[vortex_mask]))
            })
    
        return vortices

    def plot_vortex_boundaries(self, xi, yi, vortices, u_rot, v_rot, time_v, alpha_i):
        """在速度场上叠加显示PCA测量的涡旋椭圆（改进版）"""
        plt.figure(figsize=self.FIG_SIZE)
        
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
            density=5,
            arrowsize=2,
            arrowstyle='->',
            zorder=2,
        )
        # 2. 绘制每个涡旋的几何椭圆和主轴
        
        for i, vortex in enumerate(vortices):
            # 几何形状椭圆
            ellipse = Ellipse(
                xy=vortex['center'],
                width=vortex['length'],
                height=vortex['width'],
                angle=np.degrees(vortex['geo_angle']),
                edgecolor='r',
                facecolor='none',
                linestyle='--',
                linewidth=2,
                zorder=3
            )
            # print(f"涡旋 {i}: 中心={vortex['center']}, 长轴={vortex['length']:.4f}, 短轴={vortex['width']:.4f}, 角度={np.degrees(vortex['geo_angle']):.2f}°")
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
                'pink', linewidth=1.5, zorder=4
            )
            
            # 标记中心点和编号
            plt.scatter(*vortex['center'], c='r', s=50, marker='x', zorder=5)
            plt.text(vortex['center'][0], vortex['center'][1], str(i),
                    color='k', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6)
        
        # 图例和装饰
        plt.title(f'Vortex Boundaries (PCA Analysis) at t={time_v}s')
        plt.xlim(self.X_LIM)
        plt.ylim(0, 0.3)
        
        # 手动创建图例元素
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='r', linestyle='--', lw=2, label='Geometric Axis'),
            Line2D([0], [0], color='b', linestyle='--', lw=2, label='Kinematic Axis'),
            Line2D([0], [0], marker='x', color='r', lw=0, label='Vortex Center')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.savefig(os.path.join(self.output_dir, f'vortex_pca_dimensions_t{time_v}.png'), 
                   dpi=300, bbox_inches='tight')
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

    def run_analysis(self):
        """Main method to run the analysis for all time steps"""
        os.makedirs(self.output_dir, exist_ok=True)
        for time_v in self.times:
            print(f"--- Processing Instantaneous Snapshot: t={time_v} ---")
            try:
                self.process_time_step(time_v)
            except Exception as e:
                print(f"    ERROR processing t={time_v}: {e}")
        print(f"\nInstantaneous analysis complete. All results saved to: {self.output_dir}")


# ==============================================================================
#  新类：POD 时序分析
# ==============================================================================
class TurbidityCurrentPODAnalyzer:
    def __init__(self, base_analyzer):
        self.base_analyzer = base_analyzer
        self.sol = base_analyzer.sol
        self.output_dir = os.path.join(base_analyzer.output_dir, "POD_analysis_perturbation")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 使用相同的网格设置
        X, Y, _ = fluidfoam.readmesh(self.sol)
        self.xi = np.linspace(X.min(), X.max(), 750)
        self.yi = np.linspace(Y.min(), Y.max(), 200)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

    def get_perturbation_fields_for_time(self, time_v):
        """
        与process_time_step中完全一致的扰动场计算方法
        返回: (u_perturb, v_perturb) 的插值场
        """
        # 1. 读取原始数据 (与process_time_step完全一致)
        Ua_A = fluidfoam.readvector(self.sol, str(time_v), "U.a")
        alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
        beta = fluidfoam.readscalar(self.sol, str(time_v), "alpha.b")
        gradU = fluidfoam.readtensor(self.sol, str(time_v), "grad(U.a)")
        vorticity = fluidfoam.readvector(self.sol, str(time_v), "vorticity")
        gradbeta = fluidfoam.readvector(self.sol, str(time_v), "grad(alpha.b)")
        gradvorticity = fluidfoam.readtensor(self.sol, str(time_v), "grad(vorticity)")
        
        gradU_x = gradU[0]
        gradU_y = gradU[3]
        gradV_x = gradU[1]
        gradV_y = gradU[4]
        omega_z = vorticity[2]
        gradbeta_x = gradbeta[0]
        gradvorticity_x = gradvorticity[2]

        X, Y, Z = fluidfoam.readmesh(self.sol)

        # 2. 计算平均速度剖面 (与process_time_step完全一致)
        head_x = None
        for x in np.unique(X):
            mask = (X == x) & (Y >= self.base_analyzer.y_min) & (alpha_A > self.base_analyzer.alpha_threshold)
            if np.any(mask):
                head_x = x
        if head_x is None:
            print(f"Warning: No head found at t={time_v}")
            return None, None

        results = []
        x_coords = np.unique(X[(X <= head_x) & (X >= 0)])
        for xx in x_coords:
            mask = (X == xx) & (Y >= 0) & (alpha_A > 1e-5)
            if not np.any(mask):
                continue
            
            ya = Y[mask]
            ua = Ua_A[0][mask]
            alpha = np.maximum(alpha_A[mask], 0)
            
            # 使用相同的积分方法
            U, H, ALPHA, H_depth, y_crossing = self.base_analyzer.integrate_quantities(ya, ua, alpha)
            results.append({"x": xx, "U": U})

        df = pd.DataFrame(results)
        x_U_mapping = dict(zip(df['x'], df['U']))

        # 3. 计算扰动场 (与calculate_perturbation_fields完全一致)
        U_perturb = Ua_A[0].copy()
        U_mean_grid = np.zeros_like(omega_z)
        for x in x_coords:
            if x in x_U_mapping:
                mask = (X == x)
                U_mean = x_U_mapping[x]
                U_perturb[mask] = Ua_A[0][mask] - U_mean
                U_mean_grid[mask] = U_mean

        # 4. 插值到规则网格 (与interpolate_fields一致)
        fields = {
            'U_perturb': U_perturb,
            'uyi': Ua_A[1],  # y方向使用原始速度（与瞬时分析一致）
            'alpha_i': alpha_A
        }
        interpolated = self.base_analyzer.interpolate_fields(X, Y, self.xi, self.yi, fields)
        
        return interpolated['U_perturb'], interpolated['uyi']

    def run_pod_analysis(self, field_type="perturbation"):
        """运行POD分析
        
        参数:
            field_type: "perturbation"或"original"，指定分析哪种速度场
        """
        self.field_type = field_type
        
        # 获取时间步列表（您需要实现_get_time_steps方法）
        # time_steps = self._get_time_steps()  
        """运行POD分析"""
        time_steps = [...]  # 获取时间步列表
        
        # 收集所有快照
        snapshot_list = []
        for t in time_steps:
            u_perturb, v = self.get_perturbation_fields_for_time(t)
            if u_perturb is not None:
                snapshot_list.append(np.stack([u_perturb, v]))
        
        if not snapshot_list:
            print("Error: No valid snapshots available for POD analysis.")
            return

        # 转换为NumPy数组
        snapshots = np.stack(snapshot_list)  # shape: (n_times, 2, n_y, n_x)
        
        # 重塑为POD需要的格式 (空间点×时间)
        n_times, _, n_y, n_x = snapshots.shape
        snapshots_reshaped = snapshots.reshape(n_times, 2, -1).transpose(1, 2, 0)  # (2, n_points, n_times)
        
        # 对u和v分量分别进行POD
        for i, component in enumerate(['u', 'v']):
            # 计算均值并去除
            mean_field = np.mean(snapshots_reshaped[i], axis=1, keepdims=True)
            fluctuations = snapshots_reshaped[i] - mean_field
            
            # 执行SVD
            U, S, Vt = np.linalg.svd(fluctuations, full_matrices=False)
            
            # 计算能量谱等
            eigenvalues = S**2
            total_energy = np.sum(eigenvalues)
            energy_ratio = eigenvalues / total_energy
            
            # 可视化结果
            self.plot_energy_spectrum(energy_ratio, component)
            
            # 重构空间模式并保存
            spatial_modes = U.reshape(-1, n_y, n_x)
            self.save_spatial_modes(spatial_modes, component)
        
        print("POD analysis completed.")

    def plot_energy_spectrum(self, energy_ratio, component):
        """绘制能量谱图"""
        plt.figure()
        plt.plot(energy_ratio, 'o-')
        plt.title(f'POD Energy Spectrum - {component} component')
        plt.xlabel('Mode number')
        plt.ylabel('Energy ratio')
        plt.savefig(os.path.join(self.output_dir, f'pod_energy_spectrum_{component}.png'))
        plt.close()

    def save_spatial_modes(self, modes, component):
        """保存前N个空间模式"""
        for i in range(min(4, len(modes))):  # 保存前4个模式
            plt.figure(figsize=(10, 4))
            plt.imshow(modes[i], cmap='RdBu')
            plt.title(f'{component} component - POD Mode {i+1}')
            plt.colorbar()
            plt.savefig(os.path.join(self.output_dir, f'pod_mode_{i+1}_{component}.png'))
            plt.close()


# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == "__main__":
    
    # --- 步骤 1: 创建一个基础分析器实例 ---
    # 这个实例将被两种分析模式共享
    base_analyzer = TurbidityCurrentAnalyzer()
    
    # --- 步骤 2: (可选) 运行您原来的瞬时场分析 ---
    # 如果您只想看某几个时刻的详细涡结构，可以取消这部分的注释
    print("--- STARTING INSTANTANEOUS ANALYSIS ---")
    base_analyzer.run_analysis()
    print("\nInstantaneous field analysis complete.\n" + "="*50 + "\n")
    
    # --- 步骤 3: (推荐) 运行新的POD时序分析 ---
    print("--- STARTING POD ANALYSIS ---")
    pod_analyzer = TurbidityCurrentPODAnalyzer(base_analyzer)
    
    # 您可以选择分析哪个场： "perturbation" 或 "original"
    pod_analyzer.run_pod_analysis()
    print("\nPOD analysis complete.")

### 如何使用这个最终文件


