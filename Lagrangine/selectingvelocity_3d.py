import numpy as np
import pandas as pd
import fluidfoam
from fractions import Fraction

"""提取 3D 场的 z 方向平均后，在 2D 平面上按 headx 的 1/3 位置抽取 ua 的 x 方向速度剖面。"""

# A 控制从 head_x 向后的取样比例，默认 1/3
A = 1 / 4
y_min = 0
alpha_threshold = 1e-5


def get_a_identifier(a_value):
    frac = Fraction(a_value).limit_denominator()
    return f"{frac.numerator}{frac.denominator}"


def get_output_files():
    a_id = get_a_identifier(A)
    return {
        "yy": f"x{a_id}xyy.csv",
        "ua": f"x{a_id}xua.csv",
    }


def _ensure_vector_field(field, mesh_shape):
    if field.ndim == 4 and field.shape[1:] == mesh_shape:
        return field
    if field.ndim == 2 and field.shape[0] == 3 and field.shape[1] == np.prod(mesh_shape):
        return field.reshape((3,) + mesh_shape, order="C")
    if field.ndim == 1 and field.size == 3 * np.prod(mesh_shape):
        return field.reshape((3,) + mesh_shape, order="C")
    raise ValueError(f"无法把向量场重建为网格形状: {field.shape}, mesh_shape={mesh_shape}")


def _ensure_scalar_field(field, mesh_shape):
    if field.shape == mesh_shape:
        return field
    if field.ndim == 1 and field.size == np.prod(mesh_shape):
        return field.reshape(mesh_shape, order="C")
    raise ValueError(f"无法把标量场重建为网格形状: {field.shape}, mesh_shape={mesh_shape}")


def _locate_head_index(alpha_2d):
    mask_x = np.any(alpha_2d > alpha_threshold, axis=1)
    valid_x = np.where(mask_x)[0]
    if len(valid_x) == 0:
        return None
    return int(valid_x.max())


def _reshape_sorted(field, sort_idx, nx, ny, nz):
    """把 flattened 或 (ncomp, N) 的场按 sort_idx 重建为 (nx,ny,nz) 或 (ncomp,nx,ny,nz)。"""
    if field.ndim == 1:
        return field[sort_idx].reshape((nx, ny, nz), order="C")
    # vector-like e.g. (3, N)
    return field[:, sort_idx].reshape((field.shape[0], nx, ny, nz), order="C")


def _build_grid_cache(X_raw, Y_raw, Z_raw):
    """从扁平化坐标构建轴向信息和排序索引。

    返回: dict 包含 sort_idx, nx, ny, nz, x_axis_3d, y_axis_3d, z_axis_3d
    """
    x_axis = np.unique(X_raw)
    y_axis = np.unique(Y_raw)
    z_axis = np.unique(Z_raw)
    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    sort_idx = np.lexsort((Z_raw, Y_raw, X_raw))

    x3d = X_raw[sort_idx].reshape((nx, ny, nz), order="C")[:, 0, 0]
    y3d = Y_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, :, 0]
    z3d = Z_raw[sort_idx].reshape((nx, ny, nz), order="C")[0, 0, :]

    return {
        "sort_idx": sort_idx,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "x_axis_3d": x3d,
        "y_axis_3d": y3d,
        "z_axis_3d": z3d,
    }


def process_time_step(sol, time_v, X, Y, Z):
    """读取单个时间步，先做 z 方向平均，再抽取 ua-x 速度剖面。"""
    Ua = fluidfoam.readvector(sol, str(time_v), "U.a")
    alpha = fluidfoam.readscalar(sol, str(time_v), "alpha.a")

    # 如果 mesh / 场是扁平化形式（1D 或分量在前），使用重建函数
    if X.ndim == 1:
        print(f"检测到扁平化网格数据，正在重建 3D 网格和场数据...")
        grid = _build_grid_cache(X, Y, Z)
        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        sort_idx = grid["sort_idx"]

        alpha = _reshape_sorted(alpha, sort_idx, nx, ny, nz)
        Ua = _reshape_sorted(Ua, sort_idx, nx, ny, nz)

        x_axis = grid["x_axis_3d"]
        y_axis = grid["y_axis_3d"]
        z_axis = grid["z_axis_3d"]
    else:
        mesh_shape = X.shape
        Ua = _ensure_vector_field(Ua, mesh_shape)
        alpha = _ensure_scalar_field(alpha, mesh_shape)

        x_axis = X[:, 0, 0] if X.ndim == 3 else np.unique(X)
        y_axis = Y[0, :, 0] if Y.ndim == 3 else np.unique(Y)

    # 先对 z 方向平均，得到 2D (x, y) 场
    alpha_2d = np.mean(alpha, axis=2)
    ua_2d = np.mean(Ua[0], axis=2)

    head_idx = _locate_head_index(alpha_2d)
    if head_idx is None:
        print(f"Warning: No head found at t={time_v}")
        return None

    head_x = float(x_axis[head_idx])
    target_x = head_x - A * 0.3
    closest_x_idx = int(np.argmin(np.abs(x_axis - target_x)))
    closest_x = float(x_axis[closest_x_idx])

    # 取该 x 位置上的 y 向剖面线；ua 取的是 x 分量速度
    mask_y = y_axis >= y_min
    yvalue = y_axis[mask_y]
    uavalue = ua_2d[closest_x_idx, mask_y]

    return {
        "time": time_v,
        "timedimless": time_v / 0.56,
        "x": closest_x,
        "yy": yvalue.tolist(),
        "ua": uavalue.tolist(),
        "head_x": head_x,
        "target_x": target_x,
    }


def save_data(data_dict, base_path, file_prefix):
    output_files = get_output_files()
    for key, filename in output_files.items():
        rows = []
        for item in data_dict:
            row = [item["time"], item["x"]] + item.get(key, [])
            rows.append(row)

        df = pd.DataFrame(rows).transpose()
        df.to_csv(f"{base_path}{file_prefix}_{filename}", index=False, header=False)


def main():
    sol = "/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230428_2"
    X, Y, Z = fluidfoam.readmesh(sol)

    times = np.arange(5, 15, 1)
    results = []
    base_path = "/home/amber/postpro/selecting_variant/"
    file_prefix = "case230428_2"

    for time_v in times:
        result = process_time_step(sol, time_v, X, Y, Z)
        if result is not None:
            results.append(result)

    if results:
        save_data(results, base_path, file_prefix)
        print(f"数据处理完成(A={A})，结果已保存到以下文件：")
        for name in get_output_files().values():
            print(f"  - {file_prefix}_{name}")
    else:
        print("未找到有效数据")


if __name__ == "__main__":
    main()
