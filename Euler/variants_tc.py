import numpy as np
import pandas as pd
import fluidfoam
import os
from pathlib import Path
from time import time
import matplotlib.pyplot as plt

# Constants
g = 9.81
R = (3217 - 1000) / 1000
rho_0 = 1000
rho_1 = 1007.7
rho_2 = 3217
alphaKomega = 0.6
Cmu = 0.09
Sc = 1
nu = 1e-6  # Kinematic viscosity

def get_case_name(case_path):
    """Extract case name from path"""
    return Path(case_path).name

def calculate_derivatives(x, y):
    """Calculate derivatives using central differences"""
    dx = np.diff(x)
    dy = np.diff(y)
    derivatives = dy / dx
    # Use forward/backward differences at boundaries
    derivatives = np.concatenate(([derivatives[0]], 
                                 (derivatives[:-1] + derivatives[1:])/2,
                                 [derivatives[-1]]))
    return derivatives

def process_time_step(case_path, time_dir, output_dir, case_name):
    """Process data for a single time step"""
    start_time = time()
    
    # Read fields using fluidfoam
    try:
        Ua = fluidfoam.readvector(case_path, str(time_dir), "U.a")
        Ub = fluidfoam.readvector(case_path, str(time_dir), "U.b")
        alpha = fluidfoam.readscalar(case_path, str(time_dir), "alpha.a")
        kb = fluidfoam.readscalar(case_path, str(time_dir), "k.b")
        omega = fluidfoam.readscalar(case_path, str(time_dir), "omega.b")
        nut = fluidfoam.readscalar(case_path, str(time_dir), "nut.b")
        gamma = fluidfoam.readscalar(case_path, str(time_dir), "K")
        gradUb = fluidfoam.readtensor(case_path, str(time_dir), "grad(U.b)")
        grad_alpha = fluidfoam.readvector(case_path, str(time_dir), "grad(alpha.a)")
    except Exception as e:
        print(f"Error reading fields at time {time_dir}: {e}")
        return None

    # Get unique x coordinates
    x_coords = np.unique(X)
        # Get unique x coordinates

    
    # 首先找到符合条件的最大x坐标
    max_valid_x = None
    for xx in x_coords:
        mask1 = (X == xx) & (alpha > 1e-5) & (Y > 0)
        if np.any(mask1):
            if max_valid_x is None or xx > max_valid_x:
                max_valid_x = xx
    
    # Store results for each x position
    time_results = []
    # Store results for each x position

        
    for i, xx in enumerate(x_coords):
        mask = (X == xx) & (alpha > 1e-5) & (Y > 0)
        if not np.any(mask):
            continue

    # # 只处理X>0.8的坐标点
    # for i, xx in enumerate(x_coords):
    #     if xx <= 1.0:  # 跳过X<=0.8的点
    #         continue
        
    #     mask = (X == xx) & (alpha > 1e-5) & (Y > 0)
    #     if not np.any(mask):
    #         continue

    # Extract data
        ya = Y[mask]
        ua_x = Ua[0][mask]
        alpha_vals = alpha[mask]
        ub_x = Ub[0][mask]
        ub_y = Ub[1][mask]
        kb_vals = kb[mask]
        omega_vals = omega[mask]
        nut_vals = nut[mask]
        gamma_vals = gamma[mask]
        grad_dvdx = gradUb[1][mask]
        grad_dudy = gradUb[3][mask]
        grad_dudx = gradUb[0][mask]
        grad_dvdy = gradUb[4][mask]
        grad_dalphadx = grad_alpha[0][mask]
        grad_dalphady = grad_alpha[1][mask]
        
        # Sort by y-coordinate
        sort_idx = np.argsort(ya)
        ya = ya[sort_idx]
        ua_x = ua_x[sort_idx]
        alpha_vals = alpha_vals[sort_idx]
        grad_dudy = grad_dudy[sort_idx]
        
        # Calculate velocity shear stress
        S_xx = grad_dudx[sort_idx]
        S_yy = grad_dvdy[sort_idx]
        S_xy = (grad_dvdx[sort_idx] + grad_dudy) / 2
        Sij_Sij = S_xx**2 + S_yy**2 + 2 * S_xy**2
        
        # Calculate turbulent quantities
        P_k = 2 * nut_vals[sort_idx] * Sij_Sij * 1000 * (1 - alpha_vals)
        epsilon_alpharho = Cmu * kb_vals[sort_idx] * omega_vals[sort_idx] * (1 - alpha_vals) * rho_0
        
        # Calculate drag force components
        G = gamma_vals[sort_idx] * (
            (ub_x[sort_idx] - ua_x) * nut_vals[sort_idx] * grad_dalphadx[sort_idx] / (Sc * (1 - alpha_vals)) +
            (ub_y[sort_idx] - ua_x) * nut_vals[sort_idx] * grad_dalphady[sort_idx] / (Sc * (1 - alpha_vals))
        )
        
        G2 = gamma_vals[sort_idx] * (1 / np.sqrt(Sc) - 1) * 2 * alpha_vals * kb_vals[sort_idx]

        ####    find the upper limit of integration by using the positive steamwise velocity  ####
        
        # Find the front height
        sign_changes = np.where(np.diff(np.sign(ua_x)))[0]
        max_ya_crossing_index = sign_changes[np.argmax(ya[sign_changes])] + 1 if len(sign_changes) > 0 else len(ya) - 1
        y_crossing = ya[max_ya_crossing_index]
        u_crossing = ua_x[max_ya_crossing_index]

        # Vectorized integration
        ua_alpha = ua_x * alpha_vals
        integral = np.trapz(ua_alpha[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU = np.trapz(ua_x[:max_ya_crossing_index], ya[:max_ya_crossing_index])
        integralU2 = np.trapz(ua_x[:max_ya_crossing_index]**2, ya[:max_ya_crossing_index])
        integral2 = np.trapz((ua_x[:max_ya_crossing_index] * alpha_vals[:max_ya_crossing_index])**2, ya[:max_ya_crossing_index])
        
        # Calculate derived quantities
        U = integralU2 / integralU if integralU != 0 else 0
        H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0
        H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0

        
        # Depth-averaged quantities
        p_k_average = np.trapz(P_k[:max_ya_crossing_index], ya[:max_ya_crossing_index]) / H if H != 0 else 0
        epsilon_average = np.trapz(epsilon_alpharho[:max_ya_crossing_index], ya[:max_ya_crossing_index]) / H if H != 0 else 0
        G_average = np.trapz(G[:max_ya_crossing_index], ya[:max_ya_crossing_index]) / H if H != 0 else 0
        G2_average = np.trapz(G2[:max_ya_crossing_index], ya[:max_ya_crossing_index]) / H if H != 0 else 0
        

        # Find the y-coordinate where alpha crosses below 1e-5
        # 首先筛选出 y > 0.005 的点
        y_threshold = 0.005
        valid_mask = ya > y_threshold

        if np.any(valid_mask):
            # 在有效范围内寻找 alpha < 1e-5 的点
            alpha_threshold = 1e-5
            below_threshold = (alpha_vals[valid_mask] < alpha_threshold)
            
            if np.any(below_threshold):
                # 找到第一个满足条件的点的相对索引
                first_below_rel_index = np.argmax(below_threshold)
                # 转换为原始数组中的绝对索引
                valid_indices = np.where(valid_mask)[0]
                max_ya_crossing_index_alpha = valid_indices[first_below_rel_index]
                y_crossing_alpha = ya[max_ya_crossing_index_alpha]  # 修正变量名
                u_crossing_alpha = ua_x[max_ya_crossing_index_alpha]
                #print(f"Found y_crossing at {y_crossing_alpha} for xx={xx}")
            else:
                # 如果没有找到，使用有效范围内的最大y值
                max_ya_crossing_index_alpha = np.where(valid_mask)[0][-1]
                y_crossing_alpha = ya[max_ya_crossing_index_alpha]  # 修正变量名
                u_crossing_alpha = ua_x[max_ya_crossing_index_alpha]
                #print(f"No alpha < {alpha_threshold} found above y={y_threshold} for xx={xx}, using max y={y_crossing_alpha}")
        else:
            # 如果没有y>0.005的点，使用最后一个点
            max_ya_crossing_index_alpha = len(ya) - 1
            y_crossing_alpha = ya[max_ya_crossing_index_alpha]  # 修正变量名
            u_crossing_alpha = ua_x[max_ya_crossing_index_alpha]  # 无法定义

        # Vectorized integration
        ua_alpha_alpha = ua_x * alpha_vals
        integral_alpha = np.trapz(ua_alpha_alpha[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha])
        integralU_alpha = np.trapz(ua_x[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha])
        integralU2_alpha = np.trapz(ua_x[:max_ya_crossing_index_alpha]**2, ya[:max_ya_crossing_index_alpha])
        integral2_alpha = np.trapz((ua_x[:max_ya_crossing_index_alpha] * alpha_vals[:max_ya_crossing_index_alpha])**2, ya[:max_ya_crossing_index_alpha])
        
        # Calculate derived quantities
        U_alpha = integralU2_alpha / integralU_alpha if integralU_alpha != 0 else 0
        H_alpha = integral_alpha**2 / integral2_alpha if integral2_alpha != 0 else 0
        ALPHA_alpha = integral_alpha / integralU_alpha if integralU_alpha != 0 else 0
        H_depth_alpha = integralU_alpha**2 / integralU2_alpha if integralU2_alpha != 0 else 0

        
        # Depth-averaged quantities
        p_k_average_alpha = np.trapz(P_k[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0
        epsilon_average_alpha = np.trapz(epsilon_alpharho[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0
        G_average_alpha = np.trapz(G[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0
        G2_average_alpha = np.trapz(G2[:max_ya_crossing_index_alpha], ya[:max_ya_crossing_index_alpha]) / H if H != 0 else 0



        # Dimensionless numbers
        denominator = g * R * ALPHA * H
        Fr = U / np.sqrt(denominator) if denominator > 0 else np.nan
        Re = U * H / nu
        
        # Drag coefficient
        grad_Ub0 = grad_dudy[0] if len(grad_dudy) > 0 else 0
        u_star2 = grad_Ub0 * nu
        Cd = u_star2 / (U**2) if U != 0 else np.nan
        #print(np.max(x_coords))
        
        # Store results for this x position
        time_results.append({
            'time': float(time_dir),
            'xx': xx,
            'xxdimless':(max_valid_x-xx)/0.3, 
            'Fr': Fr,
            'Re': Re,
            'h_alpha': H,
            'U': U,
            'ALPHA': ALPHA,
            'h_depth': H_depth,
            'h_alpha_alpha': H_alpha,
            'U_alpha': U_alpha,
            'ALPHA_alpha': ALPHA_alpha,
            'h_depth_alpha': H_depth_alpha,
            'y_crossing': y_crossing,
            'y_crossing_alpha': y_crossing_alpha,
            'u_crossing': u_crossing,
            'u_crossing_alpha': u_crossing_alpha,
            # 'Cd': Cd,
            # 'p_k_average': p_k_average,
            # 'epsilon_average': epsilon_average,
            # 'G_average': G_average,
            # 'G2_average': G2_average,
            # 'G_average_alpha': G_average_alpha,
            # 'G2_average_alpha': G2_average_alpha,
            # 'p_k_average_alpha': p_k_average_alpha,
            # 'epsilon_average_alpha': epsilon_average_alpha
        })
    
    # Create DataFrame for current time step
    if not time_results:
        return None
        
    df = pd.DataFrame(time_results)
    
    # Calculate gradients
    if len(df) > 1:
        df_sorted = df.sort_values('xx')
        df_sorted['dydx'] = calculate_derivatives(df_sorted['xx'].values, df_sorted['h_depth'].values)
        df_sorted['duhdx'] = calculate_derivatives(df_sorted['xx'].values, 
                                                 df_sorted['U'].values * df_sorted['h_depth'].values)
        df_sorted['E'] = np.abs(df_sorted['duhdx']) / df_sorted['U']
        df = df_sorted.sort_index()  # Restore original order
    
    # Save results
    output_file = os.path.join(output_dir, f'{case_name}_results_t{time_dir}.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Processed time {time_dir} in {time()-start_time:.2f} seconds")
    return df

# Main execution
if __name__ == "__main__":
    # Setup paths
    case_path = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090429_1"
    output_dir = "/home/amber/postpro/targeted_variant"
    os.makedirs(output_dir, exist_ok=True)
    
    case_name = get_case_name(case_path)
    print(f"Processing case: {case_name}")
    
    # Read mesh once (assuming it doesn't change)
    global X, Y, Z
    X, Y, Z = fluidfoam.readmesh(case_path)
    
    # Process time steps
    time_dirs = np.arange(10.0, 11, 1)
    all_dfs = []  # Store DataFrames for all time steps
    
    for time_dir in time_dirs:
        df = process_time_step(case_path, time_dir, output_dir, case_name)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all time steps if needed
    # if all_dfs:
    #     final_df = pd.concat(all_dfs, ignore_index=True)
    #     final_output = os.path.join(output_dir, f'{case_name}_all_results.csv')
    #     final_df.to_csv(final_output, index=False)
    
    print("Processing complete!")
