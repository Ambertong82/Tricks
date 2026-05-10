import pyvista as pv
import numpy as np

# 1. 加载两个结果（请替换为你的实际文件名）
# 假设一个是细颗粒 (Fine)，一个是粗颗粒 (Coarse)
ds_fine = pv.read('/home/amber/postpro/TKE_budget/tc3d_d09_0327_1/vtk_t25.00/Ri_t25.00.vtk')
ds_coarse = pv.read('/home/amber/postpro/TKE_budget/tc3d_d23_0327_1conservation/vtk_t25.00/Ri_t25.00.vtk')

# 2. 提取数据
# 哪怕它们在文件里都叫 'Rig'，在代码里它们是独立的
rig_f = ds_fine.point_data['Ri']
rig_c = ds_coarse.point_data['Ri']

# 3. 计算比值 Ratio = Fine / Coarse
# 增加防除零保护
ratio = np.divide(rig_f, rig_c, 
                  out=np.zeros_like(rig_f), 
                  where=(np.abs(rig_c) > 1e-12))

# 4. 创建一个输出文件
# 我们以细颗粒的网格为模板，把所有数据存进去
output = ds_fine.copy()
output.point_data['rig_fine'] = rig_f    # 存入细颗粒原始值
output.point_data['rig_coarse'] = rig_c  # 存入粗颗粒原始值
output.point_data['rig_ratio'] = ratio    # 存入比值

# 5. 保存结果
output.save('Rig_Comparison_Result25.vtk')
print("✅ 处理完成！请在 ParaView 中打开 Rig_Comparison_Result.vtk")
print("你会看到三个变量：rig_fine, rig_coarse, rig_ratio")
