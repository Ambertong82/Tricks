import numpy as np
import fluidfoam
from pylab import plt, matplotlib
from matplotlib.ticker import StrMethodFormatter

matplotlib.rcParams.update({"font.size": 16})
matplotlib.rcParams["lines.linewidth"] = 3
matplotlib.rcParams["lines.markersize"] = 5
matplotlib.rcParams["lines.markeredgewidth"] = 1

##################################################################
# Parameters to retrieve dimensionless variables
##################################################################

d = 160e-6  # particle diameter in m
gravity = 9.81  # gravity in m/s2
rhoFluid = 1041  # fluid density in kg/m3
rhoSolid = 2500  # solid density in kg/m3
h = 0.0049  # initial granular height in m
theta = 25  # plane slope
val_p = (rhoSolid - rhoFluid) * gravity * h  # pressure at the bottom
timeAdim = (d / gravity) ** 0.5
velAdim = 1000.0 * (gravity * d) ** 0.5
pressureAdim = rhoFluid * h * gravity

#########################################
# Loading SedFoam results
#########################################

sol = "/media/amber/PhD_data_xtsun/PhD/Bonnecaze/Middle_particle23/case230427_4"
X, Y, Z = fluidfoam.readmesh(sol)
tolAlpha = 0.55



# this part of the script analyzes the vertical profiles at times>0 (after tilting the plane)
times = [
    5,
    6,
    7,
    
]  # please select specific times where data will be reconstructed

velocityProfiles = []
phiProfiles = []
yProfiles = []
particlePressureProfiles = []
excessPressureProfiles = []

newHeight = h
for i in range(len(times)):
    time_v = times[i]
    tread = str(times[i] + 200) + "/"
    alpha_A = fluidfoam.readscalar(sol, time_v, "alpha.a")
    Ua_A = fluidfoam.readvector(sol, time_v, "U.a")
    vel_values = []
    phi_values = []
    y_values = []
    p_particle_values = []
    vel_values = []
    p_excess_values = []
    y_values = Y



    velocityProfiles.append(vel_values)
    yProfiles.append(y_values)



#########################################
# 				Plots
#########################################


# velocity profile
plt.figure()

plt.plot(
    velocityProfiles[0],
    yProfiles[0],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="powderblue",
)
plt.plot(
    velocityProfiles[1],
    yProfiles[1],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="deepskyblue",
)
plt.plot(
    velocityProfiles[2],
    yProfiles[2],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="dodgerblue",
)
plt.plot(
    velocityProfiles[3],
    yProfiles[3],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="royalblue",
)
plt.plot(
    velocityProfiles[4],
    yProfiles[4],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="navy",
)
plt.ylabel("$y/h_o$ [$-$]", fontsize=18)
plt.xlabel("$v^s/\\sqrt{gd}$ [$-$]", fontsize=18)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))  #
plt.grid()
plt.tight_layout()
# plt.legend(prop={'size':10.0},loc=0)
plt.savefig("Figures/VelocityProfile1D_phi0592.png", dpi=200)


# volume fraction profile
plt.figure()
plt.plot(
    phi_0,
    y_0,
    marker="o",
    markersize=0,
    linestyle="--",
    linewidth=1.5,
    color="k",
    label="$t=0s$",
)
plt.plot(
    phiProfiles[0],
    yProfiles[0],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="powderblue",
)
plt.plot(
    phiProfiles[1],
    yProfiles[1],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="deepskyblue",
)
plt.plot(
    phiProfiles[2],
    yProfiles[2],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="dodgerblue",
)
plt.plot(
    phiProfiles[3],
    yProfiles[3],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="royalblue",
)
plt.plot(
    phiProfiles[4],
    yProfiles[4],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="navy",
)
plt.ylabel("$y/h_o$ [$-$]", fontsize=18)
plt.xlabel("$\\phi$ [$-$]", fontsize=18)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))
plt.grid()
plt.tight_layout()
# plt.legend(loc=0)
plt.savefig("Figures/PhiProfile1D_phi0592.png", dpi=200)


# excess of fluid pore pressure profile
plt.figure()
plt.plot(
    p_excess_0,
    y_0,
    marker="o",
    markersize=0,
    linestyle="--",
    linewidth=1.5,
    color="k",
    label="$t=0s$",
)
plt.plot(
    excessPressureProfiles[0],
    yProfiles[0],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="powderblue",
    label="$t=%s s$" % times[0],
)
plt.plot(
    excessPressureProfiles[1],
    yProfiles[1],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="deepskyblue",
    label="$t=%s s$" % times[1],
)
plt.plot(
    excessPressureProfiles[2],
    yProfiles[2],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="dodgerblue",
    label="$t=%s s$" % times[2],
)
plt.plot(
    excessPressureProfiles[3],
    yProfiles[3],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="royalblue",
    label="$t=%s s$" % times[3],
)
plt.plot(
    excessPressureProfiles[4],
    yProfiles[4],
    marker="o",
    markersize=0,
    linestyle="-",
    linewidth=1.5,
    color="navy",
    label="$t=%s s$" % times[4],
)
plt.ylabel("$y/h_o$ [$-$]", fontsize=18)
plt.xlabel("$\\frac{p^f}{(\\rho^s - \\rho^f)g h_o}$ [$-$]", fontsize=21)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
plt.grid()
plt.legend(prop={"size": 15.0}, loc=0)
plt.tight_layout()
plt.savefig("Figures/FluidPressureProfile1D_phi0592.png", dpi=200)

plt.show()