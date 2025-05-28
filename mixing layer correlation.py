import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# 1. 生成示例数据（替换为你的实际数据）
file1_path = "/home/amber/postpro/mixinglayerEuler/case230427_4_output_59.csv"
# file1_path = "/home/amber/postpro/case090429_1_1e5.csv"
df1 = pd.read_csv(file1_path)
# 2. 计算波动成分
t = df1['time'][0:]
xori = df1['x_position'][0:]
uori = df1['delta_ua'][0:]
front = df1['front_position'][0:]

H = 0.3
g = 9.81
rhos = 3217
rho = 1000
alpha = 0.011
rho1 = alpha * rhos + (1 - alpha) * rho
g = 9.81 * (rho1 - rho) / rho

xdimless = (front - xori)/ H
print(f"xdimless={xdimless}")

# 只取 xdimless 在 [2, 4] 区间的数据
fit_mask = (xdimless >= 1.6) & (xdimless <= 3.0)
x_fit = xdimless[fit_mask]
y_fit = uori[fit_mask]

# 线性拟合
if len(x_fit) > 1:
    coeffs = np.polyfit(x_fit, y_fit, 1)  # 一次多项式拟合
    y_pred = np.polyval(coeffs, x_fit)
    plt.plot(x_fit, y_pred, 'r--', linewidth=2.5, label=f'Linear fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')



# 评估指标
    residuals = y_fit - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(y_fit))
    mae = np.mean(np.abs(residuals))
    nrmse = rmse / (np.max(y_fit) - np.min(y_fit))
    
    # 在图上显示评估指标
    plt.text(0.7, 0.6,
             f'$R^2$={r_squared:.4f}\nRMSE={rmse:.4f}\nMAE={mae:.4f}\nNRMSE={nrmse:.4f}',
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))


# 原始数据
mask = xdimless > 0
plt.plot(xdimless[mask], uori[mask], label='U.a')
plt.xlabel('$(x_{front}-x)/H$')
plt.ylabel('$\Delta U$')
plt.title('Mixing Layer Correlation at t = {:.2f} s'.format(t.iloc[0]))
plt.legend()
output_name = f"/home/amber/postpro/mixinglayer_t{t.iloc[0]:.2f}s.png"
plt.savefig(output_name)
plt.show()

# # 在图上添加文本
# plt.text(
#     0.9,
#     0.1,
#     f'R² = {r_squared:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nNRMSE = {nrmse:.4f}',
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     bbox=dict(
#         facecolor='white',
#         alpha=0.5))

# plt.savefig('/home/amber/postpro/residual_analysis2.png')  # 保存图像
# plt.show()

# # Froude Number 可视化
# params, _ = curve_fit(power_law, tdim, Fr)
# C_fitFr = params[0]
# predicted_Fr = C_fitFr * tdim**(-1 / 3)
# residuals_Fr = Fr - predicted_Fr
# ss_res_Fr = np.sum(residuals_Fr**2)
# ss_tot_Fr = np.sum((Fr - np.mean(Fr))**2)
# r_squared_Fr = 1 - (ss_res_Fr / ss_tot_Fr)
# rmse_Fr = np.sqrt(ss_res_Fr / len(Fr))
# mae_Fr = np.mean(np.abs(residuals_Fr))
# nrmse_Fr = rmse_Fr / (np.max(Fr) - np.min(Fr))

# plt.scatter(tdimori, Frori, label='Froude Number')
# plt.plot(tdim, power_law(tdim, C_fitFr), 'r-',
#          label=f'Fit: $Fr = {C_fitFr:.2f} \\cdot t^{{-1/3}}$')
# plt.xlabel('$t/\\sqrt{H/g}$')
# plt.ylabel('Froude Number')
# plt.legend()

# # 在图上添加文本
# plt.text(
#     0.1,
#     0.1,
#     f'R² = {r_squared_Fr:.4f}\nRMSE = {rmse_Fr:.4f}\nNRMSE = {nrmse_Fr:.4f}',
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     bbox=dict(
#         facecolor='white',
#         alpha=0.5))

# plt.savefig('/home/amber/postpro/23froude_number_fit3.png')  # 保存图像
# plt.show()
