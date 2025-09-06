import numpy as np
import pandas as pd
import fluidfoam
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
### THis code is used to calculate the whole turbidity current characteristics at every time step #####
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

# Setup paths
#case_path = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_5"  # Replace with your case path
case_path = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"  # Replace with your case path
output_dir = "/home/amber/postpro/targeted_variant"  # Replace with your output directory
os.makedirs(output_dir, exist_ok=True)

# Read OpenFOAM mesh
X, Y, Z = fluidfoam.readmesh(case_path)

# Get available time directories
time_dirs = np.arange(1, 40, 1)
#time_dirs = sorted(time_dirs, key=lambda x: float(x))

all_results = pd.DataFrame()
x_values = []  # Store x values for gradient calculation
h_values = []  # Store h values for gradient calculation
U_values = []

for time_dir in time_dirs:
    #print(f"Processing time: {time_dir}")
    
    # Read fields using fluidfoam
    Ua = fluidfoam.readvector(case_path, str(time_dir), "U.a")  # Dispersed phase velocity
    Ub = fluidfoam.readvector(case_path, str(time_dir), "U.b")  # Continuous phase velocity
    alpha = fluidfoam.readscalar(case_path, str(time_dir), "alpha.a")  # Volume fraction
    kb = fluidfoam.readscalar(case_path, str(time_dir), "k.b")  # Turbulent kinetic energy
    omega = fluidfoam.readscalar(case_path, str(time_dir), "omega.b")  # Specific dissipation rate
    nut = fluidfoam.readscalar(case_path, str(time_dir), "nut.b")  # Turbulent viscosity
    gamma = fluidfoam.readscalar(case_path, str(time_dir), "K")  # Interface transfer coefficient
    
    # Read gradients (if available)
    try:
        gradUb = fluidfoam.readtensor(case_path, str(time_dir), "grad(U.b)")
        grad_alpha = fluidfoam.readvector(case_path, str(time_dir), "grad(alpha.a)")
    except:
        print(f"Warning: Gradient fields not found at time {time_dir}")
        continue
    
    # Get unique x coordinates (assuming 2D simulation)
    x_coords = np.unique(X)
    
    for xx in x_coords:
        # Filter points at current x-coordinate (and z=0 for 2D)
        mask = (X == xx)  & (alpha > 1e-5) & (Y > 0)
        
        if not np.any(mask):
            continue
            
        # Extract data at these points
        ya = Y[mask]
        ua_x = Ua[0][mask]
        ua_y = Ua[1][mask]
        ub_x = Ub[0][mask]
        ub_y = Ub[1][mask]
        alpha_vals = alpha[mask]
        kb_vals = kb[mask]
        omega_vals = omega[mask]
        nut_vals = nut[mask]
        gamma_vals = gamma[mask]
        grad_dvdx = gradUb[1][mask]  # dUy/dx
        grad_dudy = gradUb[3][mask]  # dUx/dy
        grad_dudx = gradUb[0][mask]  # dUx/dx
        grad_dvdy = gradUb[4][mask]  # dUy/dy
        grad_dalphadx = grad_alpha[0][mask]
        grad_dalphady = grad_alpha[1][mask]
        
        # Sort by y-coordinate
        sort_idx = np.argsort(ya)
        ya = ya[sort_idx]
        ua_x = ua_x[sort_idx]
        ua_y = ua_y[sort_idx]
        alpha_vals = alpha_vals[sort_idx]
        kb_vals = kb_vals[sort_idx]
        omega_vals = omega_vals[sort_idx]
        nut_vals = nut_vals[sort_idx]
        gamma_vals = gamma_vals[sort_idx]
        grad_dvdx = grad_dvdx[sort_idx]
        grad_dudy = grad_dudy[sort_idx]
        grad_dudx = grad_dudx[sort_idx]
        grad_dvdy = grad_dvdy[sort_idx]
        grad_dalphadx = grad_dalphadx[sort_idx]
        grad_dalphady = grad_dalphady[sort_idx]
        
        # Calculate velocity shear stress
        S_xx = grad_dudx
        S_yy = grad_dvdy
        S_xy = (grad_dvdx + grad_dudy) / 2
        Sij_Sij = S_xx**2 + S_yy**2 + 2 * S_xy**2
        
        # Calculate turbulent generation
        P_k = 2 * nut_vals * Sij_Sij * 1000 * (1 - alpha_vals)
        
        # Calculate dissipation rate
        epsilon_alpharho = Cmu * kb_vals * omega_vals * (1 - alpha_vals) * rho_0
        
        # Calculate drag force components
        G11 = gamma_vals * (ub_x - ua_x) * nut_vals * grad_dalphadx / (Sc * (1 - alpha_vals))
        G12 = gamma_vals * (ub_y - ua_y) * nut_vals * grad_dalphady / (Sc * (1 - alpha_vals))
        G = G11 + G12
        
        G2 = gamma_vals * (1 / np.sqrt(Sc) - 1) * 2 * alpha_vals * kb_vals
        
        # Find the front height (where velocity changes sign)
        sign_changes = np.where(np.diff(np.sign(ua_x)))[0]
        if len(sign_changes) > 0:
            max_ya_crossing_index = sign_changes[np.argmax(ya[sign_changes])] + 1
        else:
            max_ya_crossing_index = len(ya) - 1
        
        # Calculate differences for integration
        differences = np.diff(ya)
        differences = np.insert(differences, 0, 0)  # Pad with 0 at start
        
        # Calculate mass flux height
        ua_alpha = ua_x * alpha_vals
        sum1 = (ua_alpha[1:max_ya_crossing_index] + ua_alpha[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        sum2 = (ua_x[1:max_ya_crossing_index] + ua_x[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        
        integral = np.sum(sum1)
        integralU = np.sum(sum2)
        
        # Calculate U and H
        alpha_ua_squared = (ua_x * alpha_vals)**2
        ua_squared = ua_x**2
        
        addU = (ua_squared[1:max_ya_crossing_index] + ua_squared[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        integralU2 = np.sum(addU)
        
        addc = (alpha_ua_squared[1:max_ya_crossing_index] + alpha_ua_squared[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        integral2 = np.sum(addc)
        
        U = integralU2 / integralU if integralU != 0 else 0
        H = integral**2 / integral2 if integral2 != 0 else 0
        ALPHA = integral / integralU if integralU != 0 else 0
        H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0
        
        # Calculate depth-averaged quantities
        add_P_k = (P_k[1:max_ya_crossing_index] + P_k[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        p_k_average = np.sum(add_P_k) / H if H != 0 else 0
        
        add_epsilon = (epsilon_alpharho[1:max_ya_crossing_index] + epsilon_alpharho[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        epsilon_average = np.sum(add_epsilon) / H if H != 0 else 0
        
        add_G = (G[1:max_ya_crossing_index] + G[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        G_average = np.sum(add_G) / H if H != 0 else 0
        
        add_G2 = (G2[1:max_ya_crossing_index] + G2[:max_ya_crossing_index-1]) * differences[1:max_ya_crossing_index] / 2
        G2_average = np.sum(add_G2) / H if H != 0 else 0
        
        # Calculate Froude and Reynolds numbers
        denominator = g * R * ALPHA * H
        Fr = U / np.sqrt(denominator) if denominator > 0 else np.nan
        Re = U * H / nu
        
        # Calculate drag coefficient
        grad_Ub0 = grad_dudy[0]  # dUx/dy at bottom
        u_star2 = grad_Ub0 * nu  # tau_wall = rho*nu*dU/dy
        Cd = u_star2 / (U**2) if U != 0 else np.nan
        
        # Store results
        x_values.append(xx)
        h_values.append(H)
        U_values.append(U)
        
        all_results = pd.concat([
            all_results,
            pd.DataFrame({
                'time': [float(time_dir)],
                'xx': [xx],
                'Fr': [Fr],
                'Re': [Re],
                'h_depth': [H],
                'U': [U],
                'ALPHA': [ALPHA],
                'H_depth': [H_depth],
                'intergral': [integral],
                'integralU': [integralU],
                'Cd': [Cd],
                'p_k_average': [p_k_average],
                'epsilon_average': [epsilon_average],
                'G_average': [G_average],
                'G2_average': [G2_average]
            })
        ])
    
    # Calculate gradients after processing all points for this time
    if len(x_values) > 1:
        sorted_indices = np.argsort(x_values)
        x_sorted = np.array(x_values)[sorted_indices]
        h_sorted = np.array(h_values)[sorted_indices]
        u_sorted = np.array(U_values)[sorted_indices]
        
        dydx = np.gradient(h_sorted, x_sorted[1]-x_sorted[0])
        duhdx = np.gradient(u_sorted * h_sorted, x_sorted[1]-x_sorted[0])
        
        # Map gradients back to original order
        dydx_mapped = np.zeros_like(dydx)
        dydx_mapped[sorted_indices] = dydx
        duhdx_mapped = np.zeros_like(duhdx)
        duhdx_mapped[sorted_indices] = duhdx
        
        # Add gradient to results
        all_results['dydx'] = dydx_mapped
        all_results['duhdx'] = duhdx_mapped
        all_results['E'] = abs(duhdx_mapped) / u_sorted
    
    # Save results for this time
    output_file = os.path.join(output_dir, f'results_{time_dir}.csv')
    all_results.to_csv(output_file, index=False)
    
    # Reset for next time
    all_results = pd.DataFrame()
    x_values = []
    h_values = []
    U_values = []

print("Processing complete!")
