import pandas as pd
import numpy as np

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
# H = 0.3

# File list generation
file_list = [
    f'/home/amber/postpro/rawdata/case230427_4_{i}.csv' for i in range(19, 40)]
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
                    ub_values = point_data['U.b:0']
                    ua_valuesy = point_data['U.a:1']
                    ub_valuesy = point_data['U.b:1']
                    ya_values = point_data['Points:1']
                    alpha_values = point_data['alpha.a']
                    kb_values = point_data['k.b']
                    grad_dvdx = point_data['grad(U.b):1']
                    grad_dudy = point_data['grad(U.b):3']
                    grad_dudx = point_data['grad(U.b):0']
                    grad_dvdy = point_data['grad(U.b):4']
                    nut_values = point_data['nut.b']
                    grad_dkdx = point_data['grad(k.b):0']
                    grad_dkdy = point_data['grad(k.b):1']
                    omega_values = point_data['omega.b']
                    grad_dalphadx = point_data['grad(alpha.a):0']
                    grad_dalphady = point_data['grad(alpha.a):1']
                    # grad_dbetadx = point_data['grad(alpha.b):0']
                    # grad_dbetady = point_data['grad(alpha.b):1']
                    gamma_values = point_data['K']

                    # calculate velocity shear stress
                    S_xx = grad_dudx
                    S_yy = grad_dvdy
                    S_xy = (grad_dvdx + grad_dudy) / 2
                    Sij_Sij = S_xx**2 + S_yy**2 + 2 * S_xy**2

                    # calculate turbulent generation
                    P_k = 2 * nut_values * Sij_Sij * 1000 * (1 - alpha_values)

                    # calculate laplacian of k transportation
                    P1 = grad_dkdx * alpha_values * rho_0 * nut_values * alphaKomega
                    P2 = grad_dkdy * alpha_values * rho_0 * nut_values * alphaKomega
                    d2kdx2 = np.gradient(P1, 0.008, edge_order=2)
                    d2kdy2 = np.gradient(P2, ya_values, edge_order=2)
                    laplacian_k = d2kdx2 + d2kdy2

                    # calculate dissipation rate
                    epsilon_alpharho = Cmu * kb_values * \
                        omega_values * (1 - alpha_values) * rho_0

                    # calulate turbulent diffusion flux
                    D1 = grad_dkdx * alpha_values * rho_0 * nu
                    D2 = grad_dkdy * alpha_values * rho_0 * nu
                    Dd2kdx2 = np.gradient(D1, 0.008, edge_order=2)
                    Dd2kdy2 = np.gradient(D2, ya_values, edge_order=2)
                    D_k = Dd2kdx2 + Dd2kdy2

                    # calculate first part of drag force
                    G11 = gamma_values * \
                        (ub_values - ua_values) * nut_values * grad_dalphadx / (Sc * (1 - alpha_values))
                    G12 = gamma_values * \
                        (ub_valuesy - ua_valuesy) * nut_values * grad_dalphady / (Sc * (1 - alpha_values))
                    G = G11 + G12

                    # calculate second part of drag force
                    G2 = gamma_values * (1 / np.sqrt(Sc) - 1) * \
                        2 * alpha_values * kb_values

                    # calculate third part of drag force
                    beta_values = (1 - alpha_values)
                    grad_dbetadx = np.gradient(
                        beta_values * rho_0, 0.008, edge_order=2)
                    grad_dbetady = np.gradient(
                        beta_values * rho_0, ya_values, edge_order=2)
                    d2betadx2 = np.gradient(grad_dbetadx, 0.008, edge_order=2)
                    d2betady2 = np.gradient(
                        grad_dbetady, ya_values, edge_order=2)
                    laplacian_beta = d2betadx2 + d2betady2

                    G3 = gamma_values * (1 / np.sqrt(Sc) - 1) * rho_0 * beta_values * \
                        kb_values * nut_values * laplacian_beta / (omega_values * Sc)

                    # find the maximum value of ya_values
                    alpha_values = np.maximum(point_data['alpha.a'], 0)
                    valid_mask = (ua_values > 0) & (alpha_values > 1e-5)
                    valid_ya = ya_values[valid_mask] if valid_mask.any(
                    ) else ya_values

                    # 初始化默认积分上限（原逻辑）
                    max_ya_crossing_index = valid_ya.idxmax() if valid_mask.any() else len(ua_values) - 1

                    # 寻找速度正负交界点
                    sign_changes = np.where(np.diff(np.sign(ua_values)))[0]
                    if len(sign_changes) > 0:
                        positive_to_negative = [
                            i for i in sign_changes
                            if (i + 1 < len(ua_values))
                            and (ua_values[i] > 0)
                            and (ua_values[i + 1] < 0)
                        ]
                        if len(positive_to_negative) > 0:
                            crossing_ya = ya_values[positive_to_negative]
                            max_ya_crossing_index = positive_to_negative[np.argmax(
                                crossing_ya)]

                    # 这是通过速度大于0来判断的交界点 但是存在问题 比如当底部速度是负值的时候
                    valid_mask = (ua_values > 0) & (alpha_values > 1e-5)

                    if valid_mask.any():  # Check if any points meet conditions
                        # Filter valid points
                        valid_ya = ya_values[valid_mask]
                        index_of_max_ya = valid_ya.idxmax()

                    # Richardson number calculation
                    rho_mix = alpha_values * 3217 + \
                        (1 - alpha_values) * 1000
                    drhody = np.gradient(rho_mix, ya_values)
                    # Rig = -g * drhody / (rho_0 * grad_Ub**2)
                    # print("ya_values:", ya_values[max_ya_crossing_index])

                    # Velocity and depth calculations
                    differences = ya_values.diff()
                    differences[0] = 0

                    #### Mass Flux height #####
                    # 确保 alpha_values 非负  # 计算质量通量 alpha*v
                    ua_alpha_values = abs(ua_values * alpha_values)
                    ua_values_abs = (ua_values)
                    # if index_of_max_ya <= 1 or index_of_max_ya > len(ua_alpha_values):
                    #     print(f"Invalid index_of_max_ya: {index_of_max_ya}")
                    #     continue
                    # sum1 = (ua_alpha_values[1:] +
                    #         ua_alpha_values[:-1]) * differences / 2
                    sum1 = (ua_alpha_values[1:max_ya_crossing_index] + \
                            ua_alpha_values[:max_ya_crossing_index - 1]) * differences[1:max_ya_crossing_index] / 2
                    sum2 = (ua_values_abs[1:max_ya_crossing_index] + ua_values_abs[: max_ya_crossing_index - 1]
                            ) * differences[1:max_ya_crossing_index] / 2
                    sum1[0] = 0
                    sum2[0] = 0

                    integral = np.sum(sum1)
                    integralU = np.sum(sum2)
                    # print('integral:', integral)
                    # print('integralU:', integralU)
                    # print('integral:', integral)
                    alpha_ua_squre = (ua_values * alpha_values)**2
                    ua_squre = ua_values**2

                    addU = (ua_squre[:max_ya_crossing_index - 1] +
                            ua_squre[1:max_ya_crossing_index]) * differences[1:max_ya_crossing_index] / 2
                    # integralU2 = sum(addU[1:index_of_max_ya])
                    integralU2 = np.sum(addU)
                    addc = (alpha_ua_squre[:max_ya_crossing_index - 1] + \
                            alpha_ua_squre[1:max_ya_crossing_index]) * differences[1:max_ya_crossing_index] / 2
                    integral2 = np.sum(addc)
                    # integral2 = sum(addc[1:index_of_max_ya])
                    # print('integralU2:', integralU2)

                    U = integralU2 / integralU if integral != 0 else 0
                    H = integral**2 / integral2 if integral2 != 0 else 0
                    ALPHA = integral / integralU if integralU != 0 else 0
                    H_depth = integralU**2 / integralU2 if integralU2 != 0 else 0
                    # print("h_depth:", H_depth)
##########################################################################
########################################## 计算深度平均的TKE BUDGET #############

                    # depth-averaged turbulent generation

                    add_P_k = P_k[1:max_ya_crossing_index] + P_k[: max_ya_crossing_index -
                                                                 1] * differences[1:max_ya_crossing_index] / 2
                    integral_P_k = np.sum(add_P_k)
                    p_k_average = integral_P_k / H if H != 0 else 0
                    # print('difference:', differences)

                    # transport turbulent ##
                    add_T_k = laplacian_k[1:max_ya_crossing_index] + \
                        laplacian_k[:max_ya_crossing_index - 1] * differences[1:max_ya_crossing_index] / 2
                    integral_T_k = np.sum(add_T_k)
                    T_k_average = integral_T_k / H if H != 0 else 0

                    # diffusion turbulent ##
                    add_D_k = D_k[1:max_ya_crossing_index] + D_k[:max_ya_crossing_index -
                                                                 1] * differences[1:max_ya_crossing_index] / 2
                    integral_D_k = np.sum(add_D_k)
                    D_k_average = integral_D_k / H if H != 0 else 0

                    # turbulent dissipation rate
                    add_epsilon = epsilon_alpharho[1:max_ya_crossing_index] + \
                        epsilon_alpharho[:max_ya_crossing_index - 1] * differences[1:max_ya_crossing_index] / 2
                    integral_epsilon = np.sum(add_epsilon)
                    epsilon_average = integral_epsilon / H if H != 0 else 0

                    # turbulent drag force part 1
                    add_G = G[1:max_ya_crossing_index] + G[:max_ya_crossing_index -
                                                           1] * differences[1:max_ya_crossing_index] / 2
                    integral_G = np.sum(add_G)
                    G_average = integral_G / H if H != 0 else 0
                    # turbulent drag force part 2
                    add_G2 = G2[1:max_ya_crossing_index] + G2[:max_ya_crossing_index -
                                                              1] * differences[1:max_ya_crossing_index] / 2
                    integral_G2 = np.sum(add_G2)
                    G2_average = integral_G2 / H if H != 0 else 0
                    # turbulent drag force part 3
                    add_G3 = G3[1:max_ya_crossing_index] + G3[:max_ya_crossing_index -
                                                              1] * differences[1:max_ya_crossing_index] / 2
                    integral_G3 = np.sum(add_G3)
                    G3_average = integral_G3 / H if H != 0 else 0

                    #### KINETIC ENERGY ####
                    alphaUa = alpha_values * ua_values**2
                    alphaUa2 = (alpha_values * ua_values**2)**2
                    sum2 = (alphaUa[1:max_ya_crossing_index] +
                            alphaUa[:max_ya_crossing_index - 1]) / 2
                    sum2[0] = 0
                    differences[0] = 0
                    # sum3 = (ua_values[1:] + ua_values[:-1]) / 2
                    add2 = (alphaUa[1:max_ya_crossing_index] + alphaUa[:max_ya_crossing_index - 1]
                            ) * differences[1:max_ya_crossing_index] / 2
                    intergral2up = np.sum(add2)
                    add3 = (alphaUa2[1:max_ya_crossing_index] + alphaUa2[:max_ya_crossing_index - 1]
                            ) * differences[1:max_ya_crossing_index] / 2
                    intergral2bottom = np.sum(add3)
                    h_ke = intergral2up**2 / intergral2bottom
                    # print(h_ke)

                    # h_rho = (alpha_values[:-1] +
                    #          alpha_values[1:]) * differences / 2
                    # h_rho[0] = 0
                    # h1 = np.sum(h_rho)

                    ### 计算drag的cd数#####
                    grad_Ub0 = point_data['grad(U.b):3'][0]
                    # print('*****************')
                    # print(grad_Ub0)
                    u_star2 = grad_Ub0 / (10**6)
                    Cd = u_star2 / (U**2)

                    denominator = g * R * ALPHA * H
                    if denominator > 0:
                        Fr = U / np.sqrt(denominator)
                    else:
                        Fr = np.nan  # 或者设置为其他默认值
                    # Assuming kinematic viscosity = 1e-6 m²/s
                    Re = U * H / (1e-6)

                    # Store results
                    x_values.append(xx)
                    h_values.append(H)
                    U_values.append(U)

                    # Add to all_results DataFrame
                    all_results = pd.concat([
                        all_results,
                        pd.DataFrame({
                            'time': [t],
                            'xx': [xx],
                            'Fr': [Fr],
                            'Re': [Re],
                            'h_depth': [H],
                            'U': [U],
                            'ALPHA': [ALPHA],
                            'H_depth': [H_depth],
                            'intergral': [integral],
                            'integralU': [integralU],
                            # 'h_rho': [h1],
                            # 'Rig': [Rig],
                            'h_ke': [h_ke],
                            'Cd': [Cd],
                            'p_k_average': [p_k_average],
                            'T_k_average': [T_k_average],
                            'D_k_average': [D_k_average],
                            'epsilon_average': [epsilon_average],
                            'G_average': [G_average],
                            'G2_average': [G2_average],
                            'G3_average': [G3_average]

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
            all_results['E'] = abs(duhdx_mapped) / u_sorted

        # Save results for this file
        all_results.to_csv(
            f'/home/amber/postpro/depth_average/case230427_4_{i}.csv',
            index=False,
            columns=[
                'time',
                'xx',
                'Fr',
                'Re',
                'h_depth',
                'U',
                'ALPHA',
                'H_depth',
                'intergral',
                'integralU',
                # 'h_rho',
                # 'Rig',
                'h_ke',
                'dydx',
                'Cd',
                'duhdx',
                'E',
                'p_k_average',
                'T_k_average',
                'D_k_average',
                'epsilon_average',
                'G_average',
                'G2_average',
                'G3_average'
            ])

        # Reset for next file
        all_results = pd.DataFrame()
        x_values = []
        h_values = []
        U_values = []
