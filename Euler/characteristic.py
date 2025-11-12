import numpy as np
import pandas as pd
import fluidfoam
import os

class TurbidityCurrentAnalyzer:
    def __init__(self):
        # Configuration parameters
        self.sol = '/media/amber/53EA-E81F/PhD/case231020_5'
        self.output_dir = "/home/amber/postpro/u_3d_coarse"
        self.alpha_threshold = 1e-10
        self.y_min = 0
        self.times = list(range(1, 20))
        self.FIG_SIZE = (40, 6)
        self.X_LIM = (0.0, 1.6)
        self.Y_LIM = (0.0, 0.3)
        self.Height = 0.3
        self.colorset = 'fuchsia'
        
        # 新增：用于存储所有时间点的目标位置Sc值
        self.target_sc_data = []

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
        # integral2 = np.trapz((ua_x[:max_ya_crossing_index] * alpha_vals[:max_ya_crossing_index])**2, ya[:max_ya_crossing_index])
        
        U = integralU2 / integralU if integralU != 0 else 0
        # H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0
        H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0
        
        return U, ALPHA, H_depth, ya[max_ya_crossing_index]
    

    def integrate_quantities2(self, ya, ua_x, alpha_vals):
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
        integralalpha = np.trapz(alpha_vals[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU = np.trapz(ua_x[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        # integral2 = np.trapz((ua_x[:max_ya_crossing_index] * alpha_vals[:max_ya_crossing_index])**2, ya[:max_ya_crossing_index])
        
        U2 = integralU / ya[max_ya_crossing_index] if integralU != 0 else 0
        # H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA2 = integralalpha / ya[max_ya_crossing_index] if integralU != 0 else 0
        H_depth2 = ya[max_ya_crossing_index] 
        
        return U2, ALPHA2, H_depth2
    


    def process_time_step(self, time_v,X,Y,Z):
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
        
        select = (Z == 0.135)
        # selecting slice at Z = xxx
        X = X[select]
        Y = Y[select]
        alpha_A = alpha_A[select]
        Ua_A = Ua_A[:, select]
        beta = beta[select]
        gradU = gradU[:, select]
        vorticity = vorticity[:, select]
        gradbeta = gradbeta[:, select]
        gradvorticity = gradvorticity[:, select]

        head_x = None
        for x in np.unique(X):
            mask = (X == x) & (Y >= self.y_min) & (alpha_A > self.alpha_threshold)
            if np.any(mask):
                head_x = x
        if head_x is None:
            print(f"Warning: No head found at t={time_v}")
            return

        # Process each x coordinate
        x_coords = np.unique(X[(X <= head_x) & (X >= 0)])
        x_data = {}
        for xx in x_coords:
            mask = (X == xx) & (Y >= 0) & (alpha_A > 1e-5)
            if not np.any(mask):
                continue
            
            ya = Y[mask]
            ua = Ua_A[0][mask]
            alpha = np.maximum(alpha_A[mask], 0)
            
            # Calculate quantities
            U, ALPHA, H_depth, y_crossing = self.integrate_quantities(ya, ua, alpha)
            U2, ALPHA2, H_depth2 = self.integrate_quantities2(ya, ua, alpha)
            rhomix = (ALPHA * 3217 + (1 - ALPHA) * 1000 - 1000) / 1000  # 密度混合物characteristic density
            rhomix2 = (ALPHA2 * 3217 + (1 - ALPHA2) * 1000 - 1000) / 1000  # 密度混合物characteristic density
            x_data[xx] = {
                'U': U, 'H_depth': H_depth, 'y_crossing': y_crossing,
                'ya': ya, 'ua': ua, 'alpha': alpha, 'rhomix_mean': rhomix,'rhomix_2': rhomix2,'H_depth_2': H_depth2, 'U_2': U2,'ALPHA_2': ALPHA2
            }

        # 计算全域dU/dx
        if len(x_data) > 1:
            x_sorted = np.sort(list(x_data.keys()))
            U_all = np.array([x_data[x]['U'] for x in x_sorted])
            H_all = np.array([x_data[x]['H_depth'] for x in x_sorted])
            U_all2 = np.array([x_data[x]['U_2'] for x in x_sorted])
            dx = x_sorted[1] - x_sorted[0]
            # print(f"dx = {dx}")
            dUdx_all = np.gradient(U_all, dx)
            dUdx_all2 = np.gradient(U_all2, dx)
            dUHdx_all = np.gradient(U_all * H_all, dx)
            dUdx_map = dict(zip(x_sorted, dUdx_all))
            dUdx_map2 = dict(zip(x_sorted, dUdx_all2))
            dUdxH_map = dict(zip(x_sorted, dUHdx_all))
        else:
            dUdx_map = {list(x_data.keys())[0]: 0}
            dUdx_map2 = {list(x_data.keys())[0]: 0}

        # 计算每个剖面的Sc值
        for xx, data in x_data.items():
            dUdx = dUdx_map[xx]
            dUdx2 = dUdx_map2[xx]
            dUdxH = dUdxH_map[xx]
            if data['U'] != 0:
                data['S_sc'] = data['rhomix_mean'] * data['H_depth'] * dUdx / data['U']
                data['S_sc_2'] = data['rhomix_2'] * data['H_depth_2'] * dUdx2 / data['U_2']
                data['dUdxH']= dUdxH
            else:
                data['S_sc'] = 0
                data['S_sc_2'] = 0

        # 提取目标位置的Sc值
        target_positions = [
            ('1/4', head_x - 0.25 * 0.3),
            ('1/3', head_x - 1/3 * 0.3), 
            ('1/2', head_x - 0.5 * 0.3),
            ('1', head_x - 1.0 * 0.3)
        ]
        
        # 存储当前时间点的目标Sc值
        time_target_sc = {'Time': time_v, 'head_x': head_x}
        
        for label, target_x in target_positions:
            # 找到最接近目标x的网格点
            closest_x = min(x_coords, key=lambda x: abs(x - target_x))
            
            if closest_x in x_data:
                sc_value = x_data[closest_x]['S_sc']
                sc_value2 = x_data[closest_x]['S_sc_2']
                UH = x_data[closest_x]['dUdxH']
                time_target_sc[f'S_sc_{label}'] = sc_value
                time_target_sc[f'S_sc2_{label}'] = sc_value2
                time_target_sc[f'dUdxH_{label}'] = UH
                time_target_sc[f'x_{label}'] = closest_x  # 记录实际位置
            else:
                time_target_sc[f'S_sc_{label}'] = np.nan
                time_target_sc[f'S_sc2_{label}'] = np.nan
                time_target_sc[f'x_{label}'] = target_x
        
        # 添加到全局目标Sc数据中
        self.target_sc_data.append(time_target_sc)

        # 原有的详细结果输出（保持不变）
        results = []
        for xx in x_coords:
            if xx not in x_data:
                continue
                
            data = x_data[xx]
            
            # 提取目标位置的Sc值（用于详细结果文件）
            # target_sc_values = {}
            # target_sc_values2 = {}
            # for label, target_x in target_positions:
            #     closest_x = min(x_coords, key=lambda x: abs(x - target_x))
            #     if closest_x in x_data:
            #         target_sc_values[f'S_sc_{label}'] = x_data[closest_x]['S_sc']
            #         target_sc_values2[f'S_sc2_{label}'] = x_data[closest_x]['S_sc_2']
            #     else:
            #         target_sc_values[f'S_sc_{label}'] = np.nan
            #         target_sc_values2[f'S_sc2_{label}'] = np.nan

            result = {
                "Time": time_v,
                "x": xx,
                "U": data['U'], 
                'U_2': data['U_2'],
                "H": data['H_depth'],
                "y_crossing": data['y_crossing'],
                "S_sc_local": data['S_sc'],
                "S_sc2_local": data['S_sc_2'],
                'dUdxH': data['dUdxH'],
                # **target_sc_values
                # **target_sc_values2
            }
            results.append(result)
            
            # 存储速度零点的 (x, y)


        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, f"integration_results_t{time_v}.csv"), index=False)
        print(f"Results saved for t={time_v} (head_x={head_x:.2f}m)")

        # 打印目标位置Sc值
        print(f"\n目标位置Sc值 (t={time_v}):")
        for label in ['1/4', '1/3', '1/2', '1']:
            sc_value = time_target_sc.get(f'S_sc_{label}', np.nan)
            sc_value2 = time_target_sc.get(f'S_sc2_{label}', np.nan)
            if not np.isnan(sc_value):
                actual_x = time_target_sc.get(f'x_{label}', 'N/A')
                print(f"  {label}H位置 (x={actual_x:.3f}m): Sc = {sc_value:.6f}")
            if not np.isnan(sc_value2):
                actual_x = time_target_sc.get(f'x_{label}', 'N/A')
                print(f"  {label}H位置 (x={actual_x:.3f}m): Sc_2 = {sc_value2:.6f}")    

    def run_analysis(self):
        """Main method to run the analysis for all time steps"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 清空目标Sc数据
        self.target_sc_data = []
        X, Y, Z = fluidfoam.readmesh(self.sol)
        
        for time_v in self.times:
            print(f"\nProcessing time step: {time_v}")
            self.process_time_step(time_v,X,Y,Z)
        
        # 保存目标位置Sc值到单独的文件
        if self.target_sc_data:
            target_df = pd.DataFrame(self.target_sc_data)
            target_file = os.path.join(self.output_dir, "target_positions_sc_valuesmiddle.csv")
            target_df.to_csv(target_file, index=False)
            print(f"\n目标位置Sc值已保存到: {target_file}")
            print(f"文件包含 {len(self.target_sc_data)} 个时间点的数据")
        
        print(f"\n所有结果已保存到: {self.output_dir}")

# ... (其他代码保持不变)
if __name__ == "__main__":
    analyzer = TurbidityCurrentAnalyzer()
    analyzer.run_analysis()