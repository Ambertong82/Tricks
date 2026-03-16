 # Postprocess userguide

  ## OpenFOAM grad(U) 分量顺序 / Component ordering


对速度场 $(U=(U_x,U_y,U_z))$，OpenFOAM 的速度梯度张量通常按
    
  $G_{ij}=\frac{\partial U_i}{\partial x_j}$
    
来理解（行是速度分量 (i)，列是空间方向 (j)）。
在后处理/导出为 9 个分量时，OpenFOAM 采用“先对所有分量做 (x) 方向导数，再(y)，再 (z)”的顺序，也就是列优先（column-major）展开：
```markdown
    [dUx/dx, dUy/dx,  dUz/dx,
    dUx/dy, dUy/dy,  dUz/dy,
    dUx/dz, dUy/dz,  dUz/dz]
```

## 行优先 & 列优先
假设一个 3×3 梯度张量（按 $G_{ij}=\partial U_i/\partial x_j)$是：

 $$ 
  G=\begin{bmatrix}
  1 & 2 & 3\\
  4 & 5 & 6\\
  7 & 8 & 9
  \end{bmatrix}$$

  $$\begin{bmatrix}
  \partial U_x/\partial x & \partial U_x/\partial y & \partial U_x/\partial z\\
  \partial U_y/\partial x & \partial U_y/\partial y & \partial U_y/\partial z\\
  \partial U_z/\partial x & \partial U_z/\partial y & \partial U_z/\partial z
  \end{bmatrix}
  $$

  行优先（row-major / C-order）：逐行读出
  中文：先一整行再下一行；英文：read rows first.

  - 展开成 1D：[1, 2, 3, 4, 5, 6, 7, 8, 9]
  - 对应导数顺序：dUx/dx, dUx/dy, dUx/dz, dUy/dx, dUy/dy, dUy/dz, dUz/dx, dUz/dy, dUz/dz

  列优先（column-major / Fortran-order）：逐列读出
  中文：先一整列再下一列；英文：read columns first.

  - 展开成 1D：[1, 4, 7, 2, 5, 8, 3, 6, 9]
  - 对应导数顺序：dUx/dx, dUy/dx, dUz/dx, dUx/dy, dUy/dy, dUz/dy, dUx/dz, dUy/dz, dUz/dz

  在 numpy 里对应关系也很直接（g 是这 9 个数）：

  - 行优先数据：G = g.reshape(3,3, order="C")
  - 列优先数据：G = g.reshape(3,3, order="F")
  
 ### 在openfoam中的定义
 
  设 OpenFOAM 的速度梯度张量记为 (T)，其分量定义为：

  $$
  T_{ij}=\frac{\partial U_j}{\partial x_i}
  $$

  即：行 (i) 是求导方向 ((x,y,z))，列 (j) 是速度分量 ((u,v,w))。

  所以矩阵写作：

  $$
  T=
  \begin{bmatrix}
  \partial u/\partial x & \partial v/\partial x & \partial w/\partial x\\
  \partial u/\partial y & \partial v/\partial y & \partial w/\partial y\\
  \partial u/\partial z & \partial v/\partial z & \partial w/\partial z
  \end{bmatrix}
  $$

  对应关系：

  - $$T_{xy}=g.xy=\partial v/\partial x$$
  - $$T_{yx}=g.yx=\partial u/\partial y$$

  OpenFOAM tensor 在文件中的 9 个数顺序是：
  [
  (xx,\ xy,\ xz,\ yx,\ yy,\ yz,\ zx,\ zy,\ zz)
  ]
  这正是按行展开（row-major）。

  ———

  ### 行优先 / 列优先（仅是数据排布规则）

  给定矩阵
  $$
  A=
  \begin{bmatrix}
  1&2&3\\
  4&5&6\\
  7&8&9
  \end{bmatrix}
  $$

  - 行优先（C-order）：逐行展开([1,2,3,4,5,6,7,8,9])
  - 列优先（F-order）：逐列展开([1,4,7,2,5,8,3,6,9])

  在 NumPy 中：

  - 行优先数据：A = a.reshape(3,3, order='C')
  - 列优先数据：A = a.reshape(3,3, order='F')

之所以有两种，是因为不同语言/库历史上内存布局不同：

  - order='C'：按行连续（C/C++ 常见）
  - order='F'：按列连续（Fortran/MATLAB 常见）

  reshape 需要知道“按什么顺序把一维数据填回二维”。


  ———

  ### 对 OpenFOAM 梯度的 Python 处理

  T = arr.reshape(nCell, 3, 3, order='C')  # T[i,j] = dU_j/dx_i

  若你想改成常见 Jacobian 记法
  $$
  J_{ij}=\frac{\partial U_i}{\partial x_j},
  $$
  则

  J = np.transpose(T, (0, 2, 1))  # J = T^T (逐cell)

  ———

  一句话总结：OpenFOAM 这里用 C-order；若想用 ($\partial U_i/\partial x_j$ ) 形式，再做一次转置。

- 底层内存可看作一串连续元素（一维）。
- 区别在于二维/多维索引映射到这串元素时，按“行优先(C)”还是“列优先(F)”。
- 所以展开成一维时顺序不同，reshape 回去时也要用匹配的顺序。



### 关于如何confirm openFOAM的grad输出到底如何
https://github.com/Ambertong82/Test.git
这里有个test，输出内容可以看到究竟如何计算的

  ## Python/Numpy 还原 3×3 张量 


  OpenFOAM 的 grad(U) 张量分量定义为 T[i,j] = ∂U_j/∂x_i。
  - 因此行是求导方向 (x,y,z)，列是速度分量 (u,v,w)。
  - 例如：g.xx=du/dx，g.xy=dv/dx，g.yx=du/dy，g.yy=dv/dy。
  - 文件中 tensor 的 9 分量顺序是 (xx, xy, xz, yx, yy, yz, zx, zy, zz)，即按行展
    开。
  - 在 NumPy 还原单元张量应使用 order='C'：T = g.reshape(3,3,order='C')。
  - 若需 Jacobian 记法 J[i,j]=∂U_i/∂x_j，则 J = T.T（对每个 cell 转置）。
  - fluidfoam 常见返回形状为 (9, N)，这只是轴顺序（分量轴在前），可用 g9 =g_raw.T 变为 (N,9)。
  - fluidfoam 的 order='F'/'C' 主要影响 structured=True 时网格维度 reshape（cell排列），不改变 xx,xy,... 的物理语义。
  - 验证时应避免边界单元、保证边界条件与解析场一致，并优先在中心单元或加密网格检查。



# 关于python调用函数脚本
因为 Python 的 import 机制 会在模块搜索路径里找同名文件。

你现在在 Euler/u_vorticity copy.py 里写了 from gradient_diagnostics import compare_gradients
同目录下有 gradient_diagnostics.py
运行 u_vorticity copy.py 时，Python 会把“当前脚本目录”加入 sys.path，所以能找到这个模块并导入
简单理解就是：
from gradient_diagnostics import ... ⇔ 去当前目录找 gradient_diagnostics.py

常见规则：

同目录最简单：from xxx import yyy
子目录要么做包（加 __init__.py），要么用相对/绝对导入
文件名不要带空格、中文、连字符（模块名规范更稳）
