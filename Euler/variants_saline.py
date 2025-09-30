import numpy as np
import pandas as pd
import fluidfoam
import os
from pathlib import Path
from time import time
import matplotlib.pyplot as plt


##--------------------------------------------------------------------------------------------------------------------##
## this code is used to calculate the whole turbidity current characteristics at every time step ###
## -------------------------------------------------------------------------------------------------------------------##


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
    """从路径中提取case名称"""
    return Path(case_path).name

def calculate_derivatives(x, y):
    """计算导数，使用中心差分"""
    dx = np.diff(x)
    dy = np.diff(y)
    derivatives = dy / dx
    # 在边界使用前向/后向差分
    derivatives = np.concatenate(([derivatives[0]], 
                                 (derivatives[:-1] + derivatives[1:])/2,
                                 [derivatives[-1]]))
    return derivatives

def process_time_step(case_path, time_dir, output_dir, case_name):
    """处理单个时间步的数据"""
    start_time = time()
    
    # Read fields using fluidfoam
    try:
        Ua = fluidfoam.readvector(case_path, str(time_dir), "U")
        
        alpha = fluidfoam.readscalar(case_path, str(time_dir), "alpha.saline")
        kb = fluidfoam.readscalar(case_path, str(time_dir), "k")
        omega = fluidfoam.readscalar(case_path, str(time_dir), "omega")
        nut = fluidfoam.readscalar(case_path, str(time_dir), "nut")
        gradUb = fluidfoam.readtensor(case_path, str(time_dir), "grad(U)")
        grad_alpha = fluidfoam.readvector(case_path, str(time_dir), "grad(alpha.saline)")
    except Exception as e:
        print(f"Error reading fields at time {time_dir}: {e}")
        return None

    # Get unique x coordinates
    x_coords = np.unique(X)
    
    # 预分配结果数组
    results = {
        'time': np.full(len(x_coords), float(time_dir)),
        'xx': x_coords,
        'Fr': np.nan,
        'Re': np.nan,
        'h_depth': np.nan,
        'U': np.nan,
        'ALPHA': np.nan,
        'H_depth': np.nan,
        'intergral': np.nan,
        'integralU': np.nan,
        'Cd': np.nan,
        'p_k_average': np.nan,
        'epsilon_average': np.nan,
        'G_average': np.nan,
        'G2_average': np.nan
    }
    
    for i, xx in enumerate(x_coords):
        mask = (X == xx) & (alpha > 1e-5) & (Y > 0)
        if not np.any(mask):
            continue
            
        # Extract data
        ya = Y[mask]
        ua_x = Ua[0][mask]
        alpha_vals = alpha[mask]
        kb_vals = kb[mask]
        omega_vals = omega[mask]
        nut_vals = nut[mask]
        grad_dvdx = gradUb[1][mask]
        grad_dudy = gradUb[3][mask]
        grad_dudx = gradUb[0][mask]
        grad_dvdy = gradUb[4][mask]
        grad_dalphadx = grad_alpha[0][mask]
        grad_dalphady = grad_alpha[1][mask]
        
        # Sort by y-coordinate
        sort_idx = np.argsort(ya)
        # ya = ya[sort_idx]
        # ua_x = ua_x[sort_idx]
        # alpha_vals = alpha_vals[sort_idx]
        
        # Calculate velocity shear stress (向量化计算)
        S_xx = grad_dudx[sort_idx]
        S_yy = grad_dvdy[sort_idx]
        S_xy = (grad_dvdx[sort_idx] + grad_dudy[sort_idx]) / 2
        Sij_Sij = S_xx**2 + S_yy**2 + 2 * S_xy**2
        
        # Calculate turbulent quantities
        P_k = 2 * nut_vals[sort_idx] * Sij_Sij * 1000 * (1 - alpha_vals)
        epsilon_alpharho = Cmu * kb_vals[sort_idx] * omega_vals[sort_idx] * (1 - alpha_vals) * rho_0
        
        # Calculate drag force components
        
        
       
        
        # Find the front height
        sign_changes = np.where(np.diff(np.sign(ua_x)))[0]
        max_ya_crossing_index = sign_changes[np.argmax(ya[sign_changes])] + 1 if len(sign_changes) > 0 else len(ya) - 1
        
        # Calculate differences for integration
        differences = np.diff(ya)
        differences = np.insert(differences, 0, 0)
        
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
        
        # Dimensionless numbers
        denominator = g * R * ALPHA * H
        Fr = U / np.sqrt(denominator) if denominator > 0 else np.nan
        Re = U * H / nu
        
        # Drag coefficient
        grad_Ub0 = grad_dudy[sort_idx][0]
        u_star2 = grad_Ub0 * nu
        Cd = u_star2 / (U**2) if U != 0 else np.nan
        
        # Store results
        results['Fr'] = Fr
        results['Re'] = Re
        results['h_depth'] = H
        results['U'] = U
        results['ALPHA'] = ALPHA
        results['H_depth'] = H_depth
        results['intergral']= integral
        results['integralU'] = integralU
        results['Cd'] = Cd
        results['p_k_average'] = p_k_average
        results['epsilon_average'] = epsilon_average

    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate gradients
    if len(df) > 1:
        df = df.sort_values('xx')
        df['dydx'] = calculate_derivatives(df['xx'].values, df['h_depth'].values)
        df['duhdx'] = calculate_derivatives(df['xx'].values, df['U'].values * df['h_depth'].values)
        df['E'] = np.abs(df['duhdx']) / df['U']
        df = df.sort_index()  # 恢复原始顺序
    
    # Save results
    output_file = os.path.join(output_dir, f'{case_name}_results_t{time_dir}.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Processed time {time_dir} in {time()-start_time:.2f} seconds")
    return df

# Main execution
if __name__ == "__main__":
    # Setup paths
    case_path = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_6"
    output_dir = "/home/amber/postpro/targeted_variant"
    os.makedirs(output_dir, exist_ok=True)
    
    case_name = get_case_name(case_path)
    print(f"Processing case: {case_name}")
    
    # Read mesh once (assuming it doesn't change)
    global X, Y, Z
    X, Y, Z = fluidfoam.readmesh(case_path)
    
    # Process time steps
    time_dirs = np.arange(1, 37, 1)
    all_results = []
    
    for time_dir in time_dirs:
        df = process_time_step(case_path, time_dir, output_dir, case_name)
        if df is not None:
            all_results.append(df)
    
    # # Combine all time steps if needed
    # if all_results:
    #     final_df = pd.concat(all_results)
    #     final_output = os.path.join(output_dir, f'{case_name}_all_results.csv')
    #     final_df.to_csv(final_output, index=False)
    
    print("Processing complete!")
