import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
file_list = [
    f'/home/amber/postpro/rawdata/case230427_4_{i}.csv' for i in range(0, 79)]
# file_list=['alpha_1.csv','alpha_2.csv']
all_results = pd.DataFrame()
data_dict = []
data_dict2 = []
data_dict3 = []
data_dict4 = []
data_dict5 = []
data_dict6 = []
data_dict7 = []
data_dict8 = []
data_dict9 = []
data_dict10 = []
data_dict11 = []
A = 1 / 3

for file in file_list:
    df = pd.read_csv(file)
    # filtered_df=df[(df['alpha.a']>0.0005)&(df['alpha.a']<0.001)]
    filtered_df = df[(df['alpha.a'] > 0.00001) & (df['Points:1'] > 0)]
    if len(filtered_df) > 1:
        max_point = filtered_df['Points:0'].idxmax()
        result = filtered_df.loc[max_point, [
            'Time', 'Points:0', 'Points:1', 'alpha.a', 'U.a:0', 'U.b:0']]
        time = filtered_df.loc[max_point, 'Time']
        all_results = pd.concat([all_results, result.to_frame().T])

        selected_point1 = filtered_df.loc[max_point, 'Points:0'] - A * 0.3
        closest_point = (
            filtered_df['Points:0'] -
            selected_point1).abs().idxmin()
        x = filtered_df.loc[closest_point, 'Points:0']

        yvalue = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'Points:1']
        # zvalue=filtered_df.loc[filtered_df['Points:0']==xvalue.iloc[0], 'Points:2'].tolist()
        uavalue = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'U.a:0']
        ubvalue = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'U.b:0']
        c_value = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'alpha.a']
        grad_Ub = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'grad(U.b):3']
        grad_dvdx = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'grad(U.b):1']
        grad_dudy = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'grad(U.b):3']
        nutb = df.loc[(df['Points:0'] == x) & (
            df['Points:2'] == 0), 'nut.b']
        grad_dudy.reset_index(drop=True, inplace=True)
        yvalue.reset_index(drop=True, inplace=True)
        uavalue.reset_index(drop=True, inplace=True)
        ubvalue.reset_index(drop=True, inplace=True)
        c_value.reset_index(drop=True, inplace=True)
        grad_Ub.reset_index(drop=True, inplace=True)
        grad_dvdx.reset_index(drop=True, inplace=True)
        grad_dudy.reset_index(drop=True, inplace=True)
        nutb.reset_index(drop=True, inplace=True)
        ubh = yvalue * uavalue
        # grad_Ua = df.loc[(df['Points:0'] == x)& (df['Points:2'] == 0) ,'grad(U.a):3']
        omegaz = grad_dvdx - grad_dudy
        reynolds = nutb * (grad_dudy + grad_dvdx)
        # print(grad_dudy)
        # print(grad_dvdx)
        # print(nutb)
        # print(reynolds)
        # print(yvalue[:-1]-yvalue[1:])
        boundary = yvalue.reset_index(drop=True)
        bottom = boundary[:-1]
        top = boundary[1:]
        bottom = bottom.reset_index(drop=True)
        top = top.reset_index(drop=True)
        # print(bottom)
        # print(top)
        center = (top - bottom) / 2 + bottom
        # print(center)
        yplus = np.sqrt(1e-6 * grad_dudy[0]) * center / 1e-6
        omegaz = omegaz.tolist()
        reynolds = reynolds.tolist()

        ## calculating Richardson number ####
        rho_mix = c_value * 2217 + 1000
        drhody = np.gradient(rho_mix, yvalue)
        dalphady = np.gradient(c_value, yvalue)
        # print(drhody)
        # 计算Richardson数
        with np.errstate(divide='ignore', invalid='ignore'):
            Rig = -9.81 * drhody / (1000 * grad_Ub**2)
            Rigg = dalphady / (grad_Ub**2)
            # RigA = -9.81 * drhody / (1000 * grad_Ua**2)
            Rig = np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0)
            # RigA = np.nan_to_num(RigA, nan=0.0, posinf=0.0, neginf=0.0)
            Rigg = np.nan_to_num(Rigg, nan=0.0, posinf=0.0, neginf=0.0)

        # calculating entrainment numbers ###
        # E_ins = np.gradient(ubh, 0.008) /(uavalue)
        # rig_values = Rig.flatten().tolist() if isinstance(Rig, np.ndarray) else [float(Rig)]
        rig_values = Rig.tolist()
        # rigA_values = RigA.flatten().tolist() if isinstance(RigA, np.ndarray) else [float(RigA)]
        rigg_values = Rigg.flatten().tolist() if isinstance(
            Rigg, np.ndarray) else [float(Rigg)]
        drhody = drhody.tolist()
        grad_Ub = grad_Ub.tolist()
        ALPHA = c_value.tolist()
        yvalue = yvalue.tolist()
        uavalue = uavalue.tolist()
        ubvalue = ubvalue.tolist()
        yplus = yplus.tolist()
        # E_ins = abs(E_ins).tolist()

        data_dict.append([time] + [x] + yvalue)
        data_dict2.append([time] + [x] + uavalue)
        data_dict3.append([time] + [x] + ubvalue)
        data_dict4.append([time, x] + rig_values)
        data_dict5.append([time, x] + rigg_values)
        data_dict6.append([time, x] + omegaz)
        data_dict7.append([time, x] + reynolds)
        data_dict8.append([time, x] + grad_Ub)
        data_dict9.append([time, x] + yplus)
        data_dict10.append([time, x] + ALPHA)
        # data_dict11.append([time, x] + E_ins)

data_df = pd.DataFrame(data_dict).transpose()
data_df.to_csv(
    '/home/amber/postpro/case230427_4_x13xyy.csv',
    index=False,
    header=False)
data2_df = pd.DataFrame(data_dict2).transpose()
data2_df.to_csv(
    '/home/amber/postpro/case230427_4_x13xua.csv',
    index=False,
    header=False)
data3_df = pd.DataFrame(data_dict3).transpose()
data3_df.to_csv(
    '/home/amber/postpro/case230427_4_x13xub.csv',
    index=False,
    header=False)
data4_df = pd.DataFrame(data_dict4).transpose()
data4_df.to_csv(
    '/home/amber/postpro/case230427_4_x13xRig.csv',
    index=False,
    header=False)
data5_df = pd.DataFrame(data_dict5).transpose()
data5_df.to_csv(
    '/home/amber/postpro/case230427_4_x13xRigg.csv',
    index=False,
    header=False)
data6_df = pd.DataFrame(data_dict6).transpose()
data6_df.to_csv(
    '/home/amber/postpro/case230427_4_x13xomegaz.csv',
    index=False,
    header=False)
data7_df = pd.DataFrame(data_dict7).transpose()
data7_df.to_csv(
    '/home/amber/postpro/case230429_5_x13xReynolds[040].csv',
    index=False,
    header=False)
data8_df = pd.DataFrame(data_dict8).transpose()
data8_df.to_csv(
    '/home/amber/postpro/case230427_4_x13dudy.csv',
    index=False,
    header=False)
data9_df = pd.DataFrame(data_dict9).transpose()
data9_df.to_csv(
    '/home/amber/postpro/case230427_4_x13YPLUS.csv',
    index=False,
    header=False)
data10_df = pd.DataFrame(data_dict10).transpose()
data10_df.to_csv(
    '/home/amber/postpro/case230427_4_x13ALPHA.csv',
    index=False,
    header=False)
# data11_df = pd.DataFrame(data_dict11).transpose()
# data11_df.to_csv('/home/amber/postpro/case230427_4_x14E_ins.csv', index=False, header=False)


# import pandas as pd
# import numpy as np

# # Constants
# A = 1 / 4
# BASE_PATH = '/home/amber/postpro/'
# FILE_PREFIX = 'case230427_4'

# # Data collection setup - keys will become row labels in output
# data_categories = {
#     'yy': 'Points:1',
#     'ua': 'U.a:0',
#     'ub': 'U.b:0',
#     'Rig': None,
#     'Rigg': None,
#     'omegaz': None,
#     'drhody': None,
#     'gradUb': 'grad(U.b):3',
#     'ALPHA': 'alpha.a',
#     'E_ins': None
# }

# def calculate_derived_values(df, x, yvalue, c_value, grad_Ub, grad_Ubvx, grad_Ubuy, uavalue, ubh):
#     """Calculate all derived quantities"""
#     # Richardson numbers
#     rho_mix = c_value * 2217 + 1000
#     drhody = np.gradient(rho_mix, yvalue)
#     dalphady = np.gradient(c_value, yvalue)

#     with np.errstate(divide='ignore', invalid='ignore'):
#         Rig = -9.81 * drhody / (1000 * grad_Ub**2)
#         Rigg = dalphady / (grad_Ub**2)
#         Rig = np.nan_to_num(Rig, nan=0.0, posinf=0.0, neginf=0.0)
#         Rigg = np.nan_to_num(Rigg, nan=0.0, posinf=0.0, neginf=0.0)

#     # Other calculations
#     omegaz = grad_Ubvx - grad_Ubuy
#     E_ins = np.gradient(ubh, 0.008) / uavalue

#     return {
#         'Rig': Rig,
#         'Rigg': Rigg,
#         'omegaz': omegaz,
#         'drhody': drhody,
#         'E_ins': E_ins
#     }

# def process_file(file):
#     """Process a single data file"""
#     df = pd.read_csv(file)
#     filtered_df = df[(df['alpha.a'] > 0.00001) & (df['Points:1'] > 0)]

#     if len(filtered_df) <= 1:
#         return None

#     max_point = filtered_df['Points:0'].idxmax()
#     time = filtered_df.loc[max_point, 'Time']
#     x = filtered_df.loc[max_point, 'Points:0'] - A * 0.3
#     closest_point = (filtered_df['Points:0'] - x).abs().idxmin()
#     x = filtered_df.loc[closest_point, 'Points:0']

#     # Extract base data
#     point_data = df[(df['Points:0'] == x) & (df['Points:2'] == 0)]
#     results = {'time': time, 'x': x}

#     for key, col in data_categories.items():
#         if col:  # For direct column data
#             results[key] = point_data[col].values

#     # Calculate derived values
#     yvalue = point_data['Points:1'].values
#     c_value = point_data['alpha.a'].values
#     grad_Ub = point_data['grad(U.b):3'].values
#     grad_Ubvx = point_data['grad(U.b):1'].values
#     grad_Ubuy = point_data['grad(U.b):3'].values
#     uavalue = point_data['U.a:0'].values
#     ubh = yvalue * uavalue

#     derived = calculate_derived_values(df, x, yvalue, c_value, grad_Ub,
#                                       grad_Ubvx, grad_Ubuy, uavalue, ubh)
#     results.update(derived)

#     return results

# def save_data_in_rows(all_data):
#     """Save data in the requested format: time and x as first two rows"""
#     for key in data_categories:
#         # Prepare data matrix
#         data_matrix = []

#         # First row: time values
#         data_matrix.append(all_data['time'])

#         # Second row: x values
#         data_matrix.append(all_data['x'])

#         # Subsequent rows: variable data (one row per file)
#         for i in range(len(all_data['time'])):
#             var_data = all_data[key][i]
#             if isinstance(var_data, (np.ndarray, list)):
#                 data_matrix.append(var_data)
#             else:
#                 data_matrix.append([var_data])

#         # Convert to DataFrame and save
#         df = pd.DataFrame(data_matrix)
#         df.to_csv(f'{BASE_PATH}{FILE_PREFIX}_x14x{key}.csv',
#                  index=False, header=False)

# def main():
#     file_list = [f'{BASE_PATH}rawdata/{FILE_PREFIX}_{i}.csv' for i in range(20, 40)]
#     all_data = {key: [] for key in ['time', 'x'] + list(data_categories.keys())}

#     for file in file_list:
#         results = process_file(file)
#         if results:
#             for key in all_data:
#                 all_data[key].append(results.get(key, np.nan))

#     # Save all data files in the requested format
#     save_data_in_rows(all_data)

# if __name__ == '__main__':
#     main()
