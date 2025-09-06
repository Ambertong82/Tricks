import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class FlowAnalyzer:
    """流体分析主类，负责数据读取、处理和可视化"""
    
    def __init__(self, sol_path: str, output_dir: str):
        """初始化分析器"""
        self.sol_path = sol_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 参数设置
        self.alpha_threshold = 1e-5
        self.y_min = 0
        self.H_ref = 0.3  # 参考高度
        
        # 读取网格数据
        self.X, self.Y, _ = fluidfoam.readmesh(sol_path)
        
        # 创建规则网格
        self.xi = np.linspace(self.X.min(), self.X.max(), 750)
        self.yi = np.linspace(self.Y.min(), self.Y.max(), 1000)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)
        
        # 绘图参数
        self.fig_size = (40, 6)
        self.x_lim = (0.5, 1.8)
        self.alpha_contour_params = {
            'levels': [1e-4],
            'colors': 'black',
            'linewidths': 2,
            'linestyles': 'dashed',
            'zorder': 3
        }
        
        # 设置全局绘图样式
        plt.rcParams.update({
            'font.size': 28,
            'axes.titlesize': 28,
            'axes.labelsize': 24,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 24
        })
    
    def calculate_q_criterion(self, gradU_x: np.ndarray, gradU_y: np.ndarray, 
                            gradV_x: np.ndarray, gradV_y: np.ndarray) -> np.ndarray:
        """计算Q准则值"""
        S_xx = gradU_x
        S_xy = 0.5*(gradU_y + gradV_x)
        S_yy = gradV_y
        
        Omega_xy = 0.5*(gradV_x - gradU_y)
        
        S_norm = S_xx**2 + 2*S_xy**2 + S_yy**2
        Omega_norm = 2*Omega_xy**2
        
        return 0.5*(Omega_norm - S_norm)
    
    def integrate_flow_properties(self, ya: np.ndarray, ua: np.ndarray, 
                                alpha: np.ndarray) -> Dict[str, float]:
        """计算垂向积分属性"""
        dy = np.diff(ya, prepend=ya[0] - 0)
        valid_mask = (ua > 0) & (alpha > self.alpha_threshold)
        
        if not np.any(valid_mask):
            return None
            
        max_ya = ya[valid_mask][-1]
        integral_mask = (ya <= max_ya)
        
        # 积分计算
        ua_alpha = ua[integral_mask] * alpha[integral_mask]
        dy_integral = dy[integral_mask]
        
        # 梯形法积分
        def trapezoidal_integral(y):
            return np.sum(0.5 * (y[1:] + y[:-1]) * dy_integral[1:])
        
        integral = trapezoidal_integral(ua_alpha)
        integralU = trapezoidal_integral(ua[integral_mask])
        integralU2 = trapezoidal_integral(ua[integral_mask]**2)
        integral2 = trapezoidal_integral(ua_alpha**2)
        
        # 计算结果
        U = integralU2 / integralU if integralU != 0 else 0
        H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0
        
        return {
            "U": U,
            "H": H,
            "ALPHA": ALPHA,
            "max_ya": max_ya
        }
    
    def process_time_step(self, time_v: float) -> Optional[pd.DataFrame]:
        """处理单个时间步数据"""
        # 读取场数据
        try:
            Ua_A = fluidfoam.readvector(self.sol_path, str(time_v), "U")
            alpha_A = fluidfoam.readscalar(self.sol_path, str(time_v), "alpha.saline")
            gradU = fluidfoam.readtensor(self.sol_path, str(time_v), "grad(U)")
        except:
            print(f"Error reading data for t={time_v}")
            return None
        
        # 提取梯度分量
        gradU_x, gradU_y = gradU[0], gradU[3]  # dUx/dx, dUx/dy
        gradV_x, gradV_y = gradU[1], gradU[4]  # dUy/dx, dUy/dy
        
        # 定位头部位置
        head_x = self._find_head_position(alpha_A)
        if head_x is None:
            print(f"Warning: No head found at t={time_v}")
            return None
        
        # 处理所有x坐标
        results = []
        x_coords = np.unique(self.X[(self.X <= head_x) & (self.X >= 0)])
        
        for x in x_coords:
            mask = (self.X == x) & (self.Y >= self.y_min)
            ya, ua, alpha = self._extract_vertical_data(mask, Ua_A[0], alpha_A)
            
            if ya.size == 0:
                continue
                
            properties = self.integrate_flow_properties(ya, ua, alpha)
            if properties:
                results.append({
                    "Time": time_v,
                    "x": x,
                    **properties
                })
        
        # 保存结果
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, f"integration_results_t{time_v}.csv"), index=False)
        
        # 计算扰动速度场
        U_perturb = self._calculate_perturbation_velocity(Ua_A[0], df)
        
        # 插值到规则网格
        interp_data = self._interpolate_fields(U_perturb, Ua_A, alpha_A, gradU_x, gradU_y, gradV_x, gradV_y)
        
        # 绘图
        self._generate_plots(interp_data, head_x, time_v)
        
        print(f"Processed t={time_v} (head_x={head_x:.2f}m)")
        return df
    
    def _find_head_position(self, alpha_A: np.ndarray) -> Optional[float]:
        """定位头部位置"""
        for x in np.unique(self.X):
            mask = (self.X == x) & (self.Y >= self.y_min) & (alpha_A > self.alpha_threshold)
            if np.any(mask):
                return x
        return None
    
    def _extract_vertical_data(self, mask: np.ndarray, ua: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取垂向数据并排序"""
        ya = self.Y[mask]
        ua = ua[mask]
        alpha = np.maximum(alpha[mask], 0)
        
        sort_idx = np.argsort(ya)
        return ya[sort_idx], ua[sort_idx], alpha[sort_idx]
    
    def _calculate_perturbation_velocity(self, Ua_x: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """计算速度扰动场"""
        x_U_mapping = dict(zip(df['x'], df['U']))
        U_perturb = Ua_x.copy()
        
        for x in np.unique(self.X):
            if x in x_U_mapping:
                mask = (self.X == x)
                U_perturb[mask] = Ua_x[mask] - x_U_mapping[x]
        
        return U_perturb
    
    def _interpolate_fields(self, U_perturb: np.ndarray, Ua_A: np.ndarray, 
                          alpha_A: np.ndarray, gradU_x: np.ndarray, 
                          gradU_y: np.ndarray, gradV_x: np.ndarray, 
                          gradV_y: np.ndarray) -> Dict[str, np.ndarray]:
        """插值各场到规则网格"""
        return {
            'U_perturb_i': griddata((self.X, self.Y), U_perturb, (self.xi, self.yi), method='cubic'),
            'uxi': griddata((self.X, self.Y), Ua_A[0], (self.xi, self.yi), method='nearest'),
            'uyi': griddata((self.X, self.Y), Ua_A[1], (self.xi, self.yi), method='nearest'),
            'alpha_i': griddata((self.X, self.Y), alpha_A, (self.xi, self.yi), method='linear'),
            'Q_i': griddata(
                (self.X, self.Y), 
                self.calculate_q_criterion(gradU_x, gradU_y, gradV_x, gradV_y),
                (self.xi, self.yi), 
                method='linear'
            )
        }
    
    def _generate_plots(self, interp_data: Dict[str, np.ndarray], 
                       head_x: float, time_v: float) -> None:
        """生成所有绘图"""
        self._plot_velocity_vectors(interp_data, time_v)
        self._plot_streamlines(interp_data, time_v, 'perturbation')
        self._plot_streamlines(interp_data, time_v, 'original')
        self._plot_q_criterion(interp_data, head_x, time_v)
    
    def _plot_velocity_vectors(self, data: Dict[str, np.ndarray], time_v: float) -> None:
        """绘制速度矢量图"""
        plt.figure(figsize=self.fig_size)
        
        skip = 5
        plt.quiver(
            self.X[::skip], self.Y[::skip],
            data['U_perturb_i'][::skip], data['uyi'][::skip],
            scale=20, width=0.002, color='blue'
        )
        
        plt.contour(self.xi, self.yi, data['alpha_i'], **self.alpha_contour_params)
        self._finalize_plot(f'Velocity Vectors (t={time_v})', 'velocity_vectors', time_v)
    
    def _plot_streamlines(self, data: Dict[str, np.ndarray], 
                         time_v: float, plot_type: str) -> None:
        """绘制流线图"""
        plt.figure(figsize=self.fig_size)
        
        if plot_type == 'perturbation':
            ux, uy = data['U_perturb_i'], data['uyi']
            color_data = np.clip(ux, -0.2, 0.05)
            title = f'Perturbation Velocity Streamlines (t={time_v})'
            cbar_label = "Velocity Perturbation $U_a\'$ [m/s]"
            density = 10
            arrowsize = 3
        else:
            ux, uy = data['uxi'], data['uyi']
            color_data = ux
            title = f'Original Velocity Streamlines (t={time_v})'
            cbar_label = "Velocity Direction (Red=Positive X, Blue=Negative X)"
            density = 5
            arrowsize = 2
        
        strm = plt.streamplot(
            self.xi, self.yi, ux, uy,
            color=color_data, cmap=plt.cm.rainbow,
            linewidth=1, density=density,
            arrowsize=arrowsize, arrowstyle='->',
            zorder=2
        )
        
        plt.contour(self.xi, self.yi, data['alpha_i'], **self.alpha_contour_params)
        plt.colorbar(strm.lines, label=cbar_label)
        self._finalize_plot(title, f'{plot_type}_streamlines', time_v)
    
    def _plot_q_criterion(self, data: Dict[str, np.ndarray], 
                         head_x: float, time_v: float) -> None:
        """绘制Q准则图"""
        plt.figure(figsize=self.fig_size)
        
        Q_levels = np.linspace(-3.5, 3.5, 21)
        Q_contour = plt.contourf(
            self.xi, self.yi, data['Q_i'],
            levels=Q_levels, cmap='bwr', extend='both'
        )
        
        plt.contour(self.xi, self.yi, data['alpha_i'], **self.alpha_contour_params)
        
        # 添加头部标记
        self._add_head_markers(head_x)
        
        plt.colorbar(Q_contour, label='Q-Criterion')
        self._finalize_plot(f'Q-Criterion Contours (t={time_v})', 'Q_criterion', time_v)
    
    def _add_head_markers(self, head_x: float) -> None:
        """添加头部位置标记"""
        positions = {
            '1/6H': head_x - (1/6)*self.H_ref,
            '1/4H': head_x - 0.25*self.H_ref,
            '1/2H': head_x - 0.5*self.H_ref,
            'H': head_x - self.H_ref
        }
        
        y_text = self.yi.min() + 0.1
        for label, x_pos in positions.items():
            plt.axvline(x=x_pos, color='b', linestyle='dashdot', linewidth=1, zorder=3)
            plt.text(x_pos + 0.01, y_text, f'xf-x={label}', 
                    fontsize=15, zorder=3, color='b')
            y_text += 0.04
    
    def _finalize_plot(self, title: str, filename_prefix: str, time_v: float) -> None:
        """完成绘图并保存"""
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.xlim(*self.x_lim)
        plt.title(title)
        plt.savefig(
            os.path.join(self.output_dir, f'{filename_prefix}_t{time_v}.png'), 
            dpi=300, bbox_inches='tight'
        )
        plt.close()


def main():
    """主函数"""
    # 设置路径
    sol_path = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_4"
    output_dir = "/home/amber/postpro/u_umean_saline"
    
    # 初始化分析器
    analyzer = FlowAnalyzer(sol_path, output_dir)
    
    # 处理时间步
    time_steps = [12, 13, 14]  # 可根据需要修改
    for time_v in time_steps:
        analyzer.process_time_step(time_v)
    
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
