import pandas as pd
import numpy as np

# Constants
g = 9.81
R = (3217 - 1000) / 1000
rho_0 = 1000
rho_1 = 1007.7
rho_2 = 3217
# H = 0.3

# File list generation
file_list = [
    f'/home/amber/postpro/rawdata/case230427_4_{i}.csv' for i in range(20, 40)]
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
                    grad_Ub = point_data['grad(U.b):3']

                    # 确保数据有效性
                    
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

                        # 计算积分
                    # sum1 = (ua_alpha_values[1:max_ya_crossing_index] +
                    #         ua_alpha_values[:max_ya_crossing_index - 1]) * differences[1:max_ya_crossing_index] / 2
                    # integral = np.sum(sum1)
                    # alpha_values = np.maximum(point_data['alpha.a'], 0)
                    # ua_values = np.where(ua_values > 0, ua_values, 0)  #

                    # 找到速度从正变负的交界点
                    # sign_changes = np.where(np.diff(np.sign(ua_values)))[0]
                    # if len(sign_changes) > 0:
                    #     # 筛选出速度从正变负的点（ua_values[sign_changes] > 0 且
                    #     # ua_values[sign_changes+1] < 0）
                    #     positive_to_negative = [i for i in sign_changes
                    #                             if i + 1 < len(ua_values)
                    #                             and ua_values[i] > 0
                    #                             and ua_values[i + 1] < 0]

                    #     if len(positive_to_negative) > 0:
                    #         crossing_ya = ya_values[positive_to_negative]
                    #     # 找到 y 最大的那个交界点
                    #         max_ya_crossing_index = positive_to_negative[np.argmax(
                    #             crossing_ya)]
                    # zero_crossing_index = positive_to_negative[0]
                    # print('*********depth********************')
                    # print('positive_to_negative', positive_to_negative)
                    # print('index_of_max_ya:', max_ya_crossing_index)
                    # print('ya_values:', ya_values[max_ya_crossing_index])
                    # print('ua_values:', ua_values[max_ya_crossing_index])
                    # print('ua_values+1', ua_values[max_ya_crossing_index +
                    # 1])

                    # 这是通过速度大于0来判断的交界点 但是存在问题 比如当底部速度是负值的时候
                    valid_mask = (ua_values > 0) & (alpha_values > 1e-5)

                    if valid_mask.any():  # Check if any points meet conditions
                        # Filter valid points
                        valid_ya = ya_values[valid_mask]
                        index_of_max_ya = valid_ya.idxmax()

                    # print('*****************************')
                    # print('index_of_max_ya:', index_of_max_ya)
                    # print('ya_values:', ya_values[index_of_max_ya])
                    # print('ua_values:', ua_values[index_of_max_ya])

                    # # Combined conditions: ua > 0 AND alpha > 1e-5
                    # valid_mask = (ua_values > 0) # & (alpha_values > 5e-5)  #
                    # 找到速度大于0的点以及alpha大于1e-5的点

                    # valid_ya = ya_values[valid_mask]
                    # index_of_max_ya = valid_ya.idxmax()

                    # print('index_of_max_ya:', index_of_max_ya)

                    # Richardson number calculation
                    rho_mix = alpha_values * 3217 + \
                        (1 - alpha_values) * 1000
                    drhody = np.gradient(rho_mix, ya_values)
                    # Rig = -g * drhody / (rho_0 * grad_Ub**2)

                    # Velocity and depth calculations
                    differences = ya_values.diff()
                    differences[0] = 0

                    #### Mass Flux height #####
                    # 确保 alpha_values 非负  # 计算质量通量 alpha*v
                    ua_alpha_values = abs(ua_values * alpha_values)
                    ua_values_abs = abs(ua_values)
                    # if index_of_max_ya <= 1 or index_of_max_ya > len(ua_alpha_values):
                    #     print(f"Invalid index_of_max_ya: {index_of_max_ya}")
                    #     continue
                    # sum1 = (ua_alpha_values[1:] +
                    #         ua_alpha_values[:-1]) * differences / 2
                    sum1 = (ua_alpha_values[1:max_ya_crossing_index] + \
                            ua_alpha_values[:max_ya_crossing_index - 1]) * differences[1:max_ya_crossing_index] / 2
                    sum2 = (ua_values_abs[1:max_ya_crossing_index] + ua_values_abs[:max_ya_crossing_index - 1]
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

                    addU = (ua_squre[:max_ya_crossing_index - 1] + \
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
                    # print(ALPHA)

                    #### TKE height ####
                    # h_tke = (kb_values[:-1] +
                    #          kb_values[1:]) * differences / 2
                    # h_tke[0] = 0
                    # H_tke = np.sum(h_tke) / 0.3

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

                            'intergral': [integral],
                            'integralU': [integralU],
                            # 'h_rho': [h1],
                            # 'Rig': [Rig],
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
            all_results['E'] = abs(duhdx_mapped) / u_sorted

        # Save results for this file
        all_results.to_csv(
            f'/home/amber/postpro/depth_average/case230427_4_{i+20}.csv',
            index=False,
            columns=[
                'time',
                'xx',
                'Fr',
                'Re',
                'h_depth',
                'U',
                'ALPHA',
                'intergral',
                'integralU',
                # 'h_rho',
                # 'Rig',
                'h_ke',
                'dydx',
                'Cd',
                'duhdx',
                'E'
                ])

        # Reset for next file
        all_results = pd.DataFrame()
        x_values = []
        h_values = []
        U_values = []
