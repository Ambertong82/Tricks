import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import fluidfoam

# 示例数据路径和时间步
# sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
# sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Fine_particle9/case090912_1"
sol = "/media/amber/PhD_data_xtsun/PhD/saline/case0704_6"
timesteps = np.arange(1, 40, 1)
X, Y, Z = fluidfoam.readmesh(sol)

# 存储每个时间步的头部速度
U_front = []
valid_times = []

for time_v in timesteps:
    try:
        #Ua = fluidfoam.readvector(sol, str(time_v), "U.a")
        Ua = fluidfoam.readvector(sol, str(time_v), "U")
        # alpha = fluidfoam.readscalar(sol, str(time_v), "alpha.a")
        alpha = fluidfoam.readscalar(sol, str(time_v), "alpha.saline")
        
        # 定位头部位置（alpha.a > 1e-5 的最大 x 坐标）
        valid_mask = (alpha > 1e-5) & (Y >= 0)
        if not np.any(valid_mask):
            print(f"Warning: No head found at t={time_v}")
            continue
            
        head_x = np.max(X[valid_mask])
        head_indices = np.where((X == head_x) & valid_mask)[0]
        
        if len(head_indices) == 0:
            continue
            
        # 提取头部速度（假设 u 是速度的 x 分量）
        head_idx = head_indices[0]
        U_front.append(Ua[0][head_idx])  # Ua[0] 是 x 分量
        valid_times.append(time_v)
        
    except Exception as e:
        print(f"Error at t={time_v}: {str(e)}")
        continue

# 转换为 numpy 数组
valid_times = np.array(valid_times)
U_front = np.array(U_front)

# --- 幂律拟合 (t >= 17s) ---
mask = valid_times > 22
t_fit = valid_times[mask]
u_fit = U_front[mask]

# 过滤无效数据点
valid_fit_mask = (t_fit > 0) & (u_fit > 0)
log_t_fit = np.log(t_fit[valid_fit_mask])
log_u_fit = np.log(u_fit[valid_fit_mask])

# 检查数据是否有效
if len(log_t_fit) == 0:
    raise ValueError("No valid data points for fitting.")

# 定义线性拟合函数（固定斜率为 -1/3）
def linear_fit(log_t, log_C):
    return log_C - (1/3) * log_t

try:
    params, _ = curve_fit(linear_fit, log_t_fit, log_u_fit)
    log_C_fit = params[0]
    C_fit = np.exp(log_C_fit)
    
    # 计算预测值和 RMSE
    u_pred = C_fit * t_fit[valid_fit_mask]**(-1/3)
    rmse = np.sqrt(np.mean((u_fit[valid_fit_mask] - u_pred)**2))

    # 在现有代码的 RMSE 计算后添加：
    u_true_fit = u_fit[valid_fit_mask]
    u_pred_fit = C_fit * t_fit[valid_fit_mask]**(-1/3)

    # R²（原始数据空间）
    ss_res = np.sum((u_fit[valid_fit_mask] - u_pred_fit)**2)
    ss_tot = np.sum((u_true_fit - np.mean(u_true_fit))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.loglog(valid_times, U_front, 'bo', label='Original Data')
    plt.loglog(t_fit[valid_fit_mask], u_pred, 'r-', 
               label=f'Fit (t > 16): $u = {C_fit:.2f} \\cdot t^{{-1/3}}$')
    plt.axvline(16, color='g', linestyle='--', alpha=0.5, label='Fit Start (t=16)')
    plt.text(0.9, 0.6, 
         f'RMSE = {rmse:.4f}\n$R^2$ = {r2:.4f}', 
         transform=plt.gca().transAxes,
         fontsize=12, 
         ha='center',  # 水平居中
         va='center',  # 垂直居中
         bbox=dict(boxstyle='round', facecolor = 'white', alpha=0.8))
    plt.xlabel('Time (log scale)')
    plt.ylabel('Velocity (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.title(f'Velocity Decay Fit (RMSE = {rmse:.4f})')
    plt.savefig('velocity_decay_fit_loglog2.png', dpi=300)
    plt.show()
    
    print(f"Fitted C = {C_fit:.4f}")
    print(f"RMSE (t > 16) = {rmse:.4f}")
    print(f"R² (t > 16) = {r2:.4f}")

except Exception as e:
    print(f"Fitting failed: {str(e)}")
