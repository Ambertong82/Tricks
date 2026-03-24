
import numpy as np
import pandas as pd
import fluidfoam
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.transforms import Affine2D

class TurbidityCurrentAnalyzer:
      def __init__(self):
          self.sol = '/media/amber/PhD_TC/Turbidity_current/Bonnecaze/Middle_particle23/case230311_2'
          self.output_dir = "/home/amber/postpro/rho_mean/tc3d_d23"
          self.times = [5]
          self.y_min = 0.0
          self.X_LIM = (0.0, 2.5)
          self.Y_LIM = (0.0, 0.3)
          self.FIG_SIZE = (40, 8)
          self.threshold = 1e-2
          # Finer plotting grid helps recover small-scale structures in contour maps.
          self.NX_PLOT = 1200
          self.NY_PLOT = 1200
          self.N_LEVELS = 120
          self.alpha_ref = 0.01
          # 2D tilt controls for projection-style cloud map.
          self.TILT_SKEW_DEG = -35
          self.TILT_Y_SCALE = 0.72

      def plot_tilted_projection(self, XI, YI, AI, time_v, vmin, vmax):
          fig, ax = plt.subplots(figsize=(16, 6))

          affine = Affine2D().scale(1.0, self.TILT_Y_SCALE).skew_deg(self.TILT_SKEW_DEG, 0.0)
          x0, x1 = self.X_LIM
          y0, y1 = self.Y_LIM

          # Render as a tilted 2D projection in data coordinates.
          img = ax.imshow(
              AI,
              origin="lower",
              extent=[x0, x1, y0, y1],
              cmap="magma_r",
              vmin=vmin,
              vmax=vmax,
              interpolation="bilinear",
              transform=affine + ax.transData,
          )

          cbar = fig.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
          cbar.set_ticks(np.linspace(0.0, 1.0, 6))
          cbar.ax.tick_params(labelsize=16)
          cbar.set_label(r"$\alpha(z-mean)/0.01$", fontsize=16)

          corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
          transformed = affine.transform(corners)
          ax.set_xlim(transformed[:, 0].min(), transformed[:, 0].max())
          ax.set_ylim(transformed[:, 1].min(), transformed[:, 1].max())
          ax.set_aspect("equal", adjustable="box")
          ax.set_xlabel("x(m)", fontsize=16)
          ax.set_ylabel("y(m)", fontsize=16)
          ax.tick_params(axis="both", labelsize=14)

          fig.tight_layout()
          tilted_path = os.path.join(self.output_dir, f"alpha_zmean_t{time_v}_tilted.png")
          fig.savefig(tilted_path, dpi=300, transparent=True)
          plt.close(fig)
          print(f"Saved: {tilted_path}")

      def plot_zmean_alpha(self, X, Y, Z, alpha, time_v):
          # 1) 组织数据
          df = pd.DataFrame({
              "x": X,
              "y": Y,
              "z": Z,
              "alpha": alpha
          })
          df = df[df["y"] >= self.y_min].copy()

          # 2) 对(x,y)做z向平均
          # 如果浮点精度有微小误差，先round一下再groupby
          df["xg"] = df["x"].round(6)
          df["yg"] = df["y"].round(6)

          df2 = df.groupby(["xg", "yg"], as_index=False)["alpha"].mean()
          df2.rename(columns={"xg": "x", "yg": "y", "alpha": "alpha_zmean"},
  inplace=True)
          # Nondimensionalize alpha by reference concentration.
          df2["alpha_nd"] = df2["alpha_zmean"] / self.alpha_ref

          # 保存z平均后的2D数据
          os.makedirs(self.output_dir, exist_ok=True)
          csv_path = os.path.join(self.output_dir, f"alpha_zmean_t{time_v}.csv")
          df2.to_csv(csv_path, index=False)

          # 3) 插值到规则网格并画云图
          xi = np.linspace(self.X_LIM[0], self.X_LIM[1], self.NX_PLOT)
          yi = np.linspace(self.Y_LIM[0], self.Y_LIM[1], self.NY_PLOT)
          XI, YI = np.meshgrid(xi, yi)

          AI_linear = griddata(
              (df2["x"].values, df2["y"].values),
              df2["alpha_nd"].values,
              (XI, YI),
              method="linear"
          )

          # Linear interpolation keeps smooth structures; nearest fills convex-hull holes only.
          AI_nearest = griddata(
              (df2["x"].values, df2["y"].values),
              df2["alpha_nd"].values,
              (XI, YI),
              method="nearest"
          )
          AI = np.where(np.isnan(AI_linear), AI_nearest, AI_linear)
          AI = np.clip(AI, 0.0, 1.0)

          valid = np.isfinite(AI)
          if not np.any(valid):
              raise ValueError("Interpolated field is all NaN; please check input mesh/data range.")

          # Fix display range to [0, 1] so colorbar is consistent across figures.
          vmin, vmax = 0.0, 1.0



          plt.figure(figsize=self.FIG_SIZE)
  


          cf = plt.contourf(
              XI,
              YI,
              AI,
              levels=np.linspace(vmin, vmax, self.N_LEVELS),
              cmap="magma_r",
              extend="max"
          )

          cbar = plt.colorbar(cf, label="alpha(z-mean)/0.01")
          cbar.set_ticks(np.linspace(0.0, 1.0, 6))
          cbar.ax.tick_params(labelsize=22)
          cbar.set_label(r'$\alpha(z-mean)/0.01$', fontsize=22)
        #   plt.contour(XI, YI, AI, levels=[1e-5], colors="k", linewidths=1.5, linestyles="--")
          plt.xlim(*self.X_LIM)
          plt.ylim(*self.Y_LIM)
          plt.xlabel("x(m)", fontsize=22)
          plt.ylabel("y(m)", fontsize=22)
          plt.xticks(fontsize=22)
          plt.yticks(fontsize=22)
          plt.gca().set_aspect('equal', adjustable='box')
          plt.tight_layout()

          fig_path = os.path.join(self.output_dir, f"alpha_zmean_t{time_v}.png")
          plt.savefig(fig_path, dpi=300)
          plt.close()

          self.plot_tilted_projection(XI, YI, AI, time_v, vmin, vmax)

          
          print(f"Saved: {csv_path}")
          print(f"Saved: {fig_path}")

      def run_analysis(self):
          X, Y, Z = fluidfoam.readmesh(self.sol)
          for time_v in self.times:
              alpha_A = fluidfoam.readscalar(self.sol, str(time_v), "alpha.a")
              self.plot_zmean_alpha(X, Y, Z, alpha_A, time_v)

if __name__ == "__main__":
      TurbidityCurrentAnalyzer().run_analysis()
