import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# 1. 生成示例数据（替换为你的实际数据）
file1_path = "/home/amber/postpro/case230427_4_1e5.csv"
#file1_path = "/home/amber/postpro/case090429_1_1e5.csv"
df1 = pd.read_csv(file1_path)
# 2. 计算波动成分
t = df1['Time'][13:]
tori = df1['Time'][0:]
u = df1['U.b:0'][13:]
uori = df1['U.a:0'][0:]
u_centered = u - np.mean(u)
H = 0.3
g = 9.81
rhos = 3217
rho = 1000
alpha = 0.011
rho1 = alpha * rhos + (1 - alpha) * rho
g = 9.81 * (rho1 - rho) / rho
Frori = uori / np.sqrt(g * H)
tdimori = tori / np.sqrt(H / g)
tdim = t / np.sqrt(H / g)
Fr = u / np.sqrt(g * H)
ufaa = uori / np.sqrt(H)

# 非线性拟合
def power_law(t, C):
    return C * t**(-1/3)

params, _ = curve_fit(power_law, t, u)
C_fit = params[0]

# 可视化
plt.scatter(tori, uori, label='U.a')
plt.plot(tori, ufaa, 'r-', label=f'Fit: $u(t) = {C_fit:.2f} \\cdot t^{{-1/3}}$')
plt.xlabel('Time (t)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.savefig('/home/amber/postpro/fit_velocity.png')  # 保存图像
plt.show()

# 计算残差和评估指标
predicted_u = C_fit * t**(-1/3)
residuals = u - predicted_u
ss_res = np.sum(residuals**2)
ss_tot = np.sum((u - np.mean(u))**2)
r_squared = 1 - (ss_res / ss_tot)
rmse = np.sqrt(ss_res / len(u))
mae = np.mean(np.abs(residuals))
nrmse = rmse / (np.max(u) - np.min(u))

# 可视化残差并显示评估指标
plt.scatter(t, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residual Analysis')

# 在图上添加文本
plt.text(0.9, 0.1, f'R² = {r_squared:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nNRMSE = {nrmse:.4f}',
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.savefig('/home/amber/postpro/residual_analysis2.png')  # 保存图像
plt.show()

# Froude Number 可视化
params, _ = curve_fit(power_law, tdim, Fr)
C_fitFr = params[0]
predicted_Fr = C_fitFr * tdim**(-1/3)
residuals_Fr = Fr - predicted_Fr
ss_res_Fr = np.sum(residuals_Fr**2)
ss_tot_Fr = np.sum((Fr - np.mean(Fr))**2)
r_squared_Fr = 1 - (ss_res_Fr / ss_tot_Fr)
rmse_Fr = np.sqrt(ss_res_Fr / len(Fr))
mae_Fr = np.mean(np.abs(residuals_Fr))
nrmse_Fr = rmse_Fr / (np.max(Fr) - np.min(Fr))

plt.scatter(tdimori, Frori, label='Froude Number')
plt.plot(tdim, power_law(tdim, C_fitFr), 'r-', label=f'Fit: $Fr = {C_fitFr:.2f} \\cdot t^{{-1/3}}$')
plt.xlabel('$t/\\sqrt{H/g}$')
plt.ylabel('Froude Number')
plt.legend()

# 在图上添加文本
plt.text(0.1, 0.1, f'R² = {r_squared_Fr:.4f}\nRMSE = {rmse_Fr:.4f}\nNRMSE = {nrmse_Fr:.4f}',
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.savefig('/home/amber/postpro/23froude_number_fit3.png')  # 保存图像
plt.show()