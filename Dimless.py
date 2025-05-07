import pandas as pd
import numpy as np

# Constants
g = 9.81
R = (3217 - 1000) / 1000
rho_0 = 1000
rho_1 = 1007.7
rho_2 = 3217
H = 0.3

# File list generation
file_list = [
    f'/home/amber/postpro/rawdata/case230427_4_{i}.csv' for i in range(21, 22)]
all_results = pd.DataFrame()
x_values = []  # Store x values for gradient calculation
h_values = []  # Store h values for gradient calculation
U_values = []

for i, file in enumerate(file_list):
    df = pd.read_csv(file)
    x0 = df['Points:0'].unique()  # Use unique() to avoid duplicate processing

    visited_xx = set()
    filtered_df = df[(df['alpha.a'] > 1e-5) & (df['Points:1'] > 0)].copy()

    if len(filtered_df) > 1:
        max_x_value = filtered_df['Points:0'].max()  # Directly get max x value

        for xx in x0:
            if 0 < xx <= max_x_value and xx not in visited_xx:
                visited_xx.add(xx)

                # Filter for current x-coordinate at z=0
                point_data = df[(df['Points:0'] == xx) &
                                (df['Points:2'] == 0)].copy()
                point_data.reset_index(drop=True, inplace=True)

                if len(point_data) > 0:
                    t = point_data.at[2, 'Time'] if len(
                        point_data) > 2 else np.nan
                    ua_values = point_data['U.a:0']
                    ya_values = point_data['Points:1']
                    alpha_values = point_data['alpha.a']
                    kb_values = point_data['k.b']

                    # Combined conditions: ua > 0 AND alpha > 1e-5
                    valid_mask = (ua_values > 0) & (alpha_values > 1e-5)

                    if valid_mask.any():  # Check if any points meet conditions
                        # Filter valid points
                        valid_ya = ya_values[valid_mask]
                        index_of_max_ya = valid_ya.idxmax()

                        # Verify conditions
                        assert (ua_values[index_of_max_ya] > 0 and
                                alpha_values[index_of_max_ya] > 1e-5), "条件不满足！"

                        # Calculate parameters
                        alpha_values = point_data['alpha.a']
                        grad_Ub = point_data['grad(U.b):3']

                        # Richardson number calculation
                        rho_mix = alpha_values * 3217 + \
                            (1 - alpha_values) * 1000
                        drhody = np.gradient(rho_mix, ya_values)
                        Rig = -g * drhody / (rho_0 * grad_Ub**2)

                        # Velocity and depth calculations
                        differences = ya_values.diff()
                        differences[0] = 0

                        #### Mass Flux height #####
                        ua_alpha_values = ua_values * alpha_values

                        sum1 = (ua_alpha_values[1:] + ua_alpha_values[:-1]) / 2
                        sum2 = (ua_values[1:] + ua_values[:-1]
                                ) * differences / 2
                        sum1[0] = 0
                        sum2[0] = 0

                        addd = differences * sum1
                        integral = sum(addd[1:index_of_max_ya + 1])
                        integralU = sum(sum2[1:index_of_max_ya + 1])
                        alpha_ua_squre = (ua_values * alpha_values)**2
                        ua_squre = ua_values**2

                        addU = (ua_squre[:-1] + ua_squre[1:]) * differences / 2
                        integralU2 = sum(addU[1:index_of_max_ya + 1])
                        addc = (
                            alpha_ua_squre[:-1] + alpha_ua_squre[1:]) * differences / 2
                        integral2 = sum(addc[1:index_of_max_ya + 1])

                        U = integralU2 / integralU if integral != 0 else 0
                        h = integral**2 / integral2 if integral2 != 0 else 0
                        ALPHA = integral / integralU if integralU != 0 else 0

                        #### TKE height ####
                        h_tke = (kb_values[:-1] +
                                 kb_values[1:]) * differences / 2
                        h_tke[0] = 0
                        H_tke = np.sum(h_tke) / 0.3

                        #### KINETIC ENERGY ####
                        alphaUa = alpha_values * ua_values**2
                        alphaUa2 = (alpha_values * ua_values**2)**2
                        sum2 = (alphaUa[1:] + alphaUa[:-1]) / 2
                        sum2[0] = 0
                        differences[0] = 0
                        # sum3 = (ua_values[1:] + ua_values[:-1]) / 2
                        add2 = (alphaUa[1:] + alphaUa[:-1]) * differences / 2
                        intergral2up = np.sum(add2)
                        add3 = (alphaUa2[1:] + alphaUa2[:-1]) * differences / 2
                        intergral2bottom = np.sum(add3)
                        h_ke = intergral2up**2 / intergral2bottom
                        # print(h_ke)

                        h_rho = (alpha_values[:-1] +
                                 alpha_values[1:]) * differences / 2
                        h_rho[0] = 0
                        h1 = np.sum(h_rho)

                        ### 计算drag的cd数#####
                        grad_Ub0 = point_data['grad(U.b):3'][0]
                        # print('*****************')
                        # print(grad_Ub0)
                        u_star2 = grad_Ub0 / (10**6)
                        Cd = u_star2 / (U**2)

                        Frd = U / np.sqrt(g * R * ALPHA * h)
                        # Assuming kinematic viscosity = 1e-6 m²/s
                        Re = U * h / (1e-6)

                        # Store results
                        x_values.append(xx)
                        h_values.append(h)
                        U_values.append(U)

                        # Add to all_results DataFrame
                        all_results = pd.concat([
                            all_results,
                            pd.DataFrame({
                                'time': [t],
                                'xx': [xx],
                                'Frd': [Frd],
                                'Re': [Re],
                                'h_depth': [h],
                                'U': [U],
                                'h_rho': [h1],
                                'Rig': [Rig[index_of_max_ya]] if len(Rig) > index_of_max_ya else [np.nan],
                                'h_ke': [h_ke],
                                'Cd': [Cd]

                            })
                        ])

        # Calculate gradient after processing all points for this file
        if len(x_values) > 1:
            sorted_indices = np.argsort(x_values)
            x_sorted = np.array(x_values)[sorted_indices]
            h_sorted = np.array(h_values)[sorted_indices]
            u_sorted = np.array(U_values)[sorted_indices]
            dydx = np.gradient(h_sorted, 0.008)  # Using fixed step 0.008
            duhdx = np.gradient(u_sorted * h_sorted, 0.008)
            # Map gradients back to original order
            dydx_mapped = np.zeros_like(dydx)
            dydx_mapped[sorted_indices] = dydx
            duhdx_mapped = np.zeros_like(duhdx)
            duhdx_mapped[sorted_indices] = duhdx
            # Add gradient to results
            all_results['dydx'] = dydx_mapped
            all_results['duhdx'] = duhdx_mapped

        # Save results for this file
        all_results.to_csv(
            f'/home/amber/postpro/depth_average/case230427_4_{i+20}.csv',
            index=False,
            columns=[
                'time',
                'xx',
                'Frd',
                'Re',
                'h_depth',
                'U',
                'h_rho',
                'Rig',
                'h_ke',
                'dydx',
                'Cd',
                'duhdx'])

        # Reset for next file
        all_results = pd.DataFrame()
        x_values = []
        h_values = []
