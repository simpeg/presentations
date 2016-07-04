from SimPEG import Mesh, Utils, Maps, Survey
from SimPEG.EM.Static import DC, IP
from pymatsolver import MumpsSolver
# from ipywidgets import interact, IntSlider
# import pickle
import sys
sys.path.append("../utilcodes/")
from vizutils import gettopoCC
# %pylab inline
import matplotlib
matplotlib.rcParams['font.size'] = 14

# Mesh and model
csx, csy, csz = 25., 25., 25.
ncx, ncy, ncz = 48, 48, 20
npad = 7
hx = [(csx,npad, -1.3),(csx,ncx),(csx,npad, 1.3)]
hy = [(csy,npad, -1.3),(csy,ncy),(csy,npad, 1.3)]
hz = [(csz,npad, -1.3),(csz,ncz), (csz/2.,6)]
mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCN")
sigma = mesh.readModelUBC("../VTKout_DC.dat")
airind = sigma == 1e-8

# Obtain topograhic surface
mesh2D, topoCC = gettopoCC(mesh, airind)

# Set TxLocs
Aloc1_x = np.r_[-600., 0, 0.]
Bloc1_x = np.r_[600., 0, 0.]


# Set RxLocs
x = mesh.vectorCCx[np.logical_and(mesh.vectorCCx>-300., mesh.vectorCCx<300.)]
y = mesh.vectorCCy[np.logical_and(mesh.vectorCCy>-300., mesh.vectorCCy<300.)]
Mx = Utils.ndgrid(x[:-1], y, np.r_[-12.5/2.])
Nx = Utils.ndgrid(x[1:], y, np.r_[-12.5/2.])
My = Utils.ndgrid(x, y[:-1], np.r_[-12.5/2.])
Ny = Utils.ndgrid(x, y[1:], np.r_[-12.5/2.])

inds_Mx = Utils.closestPoints(mesh2D, Mx[:,:2])
inds_Nx = Utils.closestPoints(mesh2D, Nx[:,:2])
inds_My = Utils.closestPoints(mesh2D, My[:,:2])
inds_Ny = Utils.closestPoints(mesh2D, Ny[:,:2])

Mx_dr = np.c_[Mx[:,0], Mx[:,1], topoCC[inds_Mx]]
Nx_dr = np.c_[Nx[:,0], Nx[:,1], topoCC[inds_Nx]]
My_dr = np.c_[My[:,0], My[:,1], topoCC[inds_My]]
Ny_dr = np.c_[Ny[:,0], Ny[:,1], topoCC[inds_Ny]]


# Src and Rx classes for DC problem
rx_x = DC.Rx.Dipole(Mx_dr, Nx_dr)
rx_y = DC.Rx.Dipole(My_dr, Ny_dr)

src1 = DC.Src.Dipole([rx_x,rx_y], Aloc1_x, Bloc1_x)


# Inversion model is log conductivity in the subsurface
# This can be realized as a following mapping:
expmap = Maps.ExpMap(mesh) # from log conductivity to conductivity
actmap = Maps.InjectActiveCells(mesh, ~airind, np.log(1e-8)) # from subsurface cells to full3D cells
mapping = expmap*actmap


# Inital and reference model
m0 = np.ones_like(sigma)[~airind]*np.log(1e-4)


# %%time
# Form survey object using Srcs and Rxs that we have generated
survey = DC.Survey([src1])
# survey = DC.Survey([src1, src2, src3, src4])
problem = DC.Problem3D_CC(mesh, mapping=mapping)
problem.Solver = MumpsSolver
problem.pair(survey)
mtrue = np.log(sigma)[~airind]
f = problem.fields(mtrue)
# Run fwd
dobs = survey.dpred(mtrue, f=f)


# Depth weighting
depth = 1./(abs(mesh.gridCC[:,2]))**1.5
depth = depth/depth.max()


from SimPEG import DataMisfit, Regularization, Optimization, Directives, InvProblem, Inversion
regmap = Maps.IdentityMap(nP=m0.size)
std = 0.05
eps = 1e-3
survey.std = std
survey.eps = eps
#TODO put warning when dobs is not set!
survey.dobs = dobs
dmisfit = DataMisfit.l2_DataMisfit(survey)
reg = Regularization.Simple(mesh, mapping=regmap, indActive=~airind)
# reg.wght = depth[~airind]
opt = Optimization.InexactGaussNewton(maxIter = 20)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
# Create an inversion object
beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1**1.5)
save = Directives.SaveOutputEveryIteration()
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta,betaest, save, target])
reg.alpha_s = 1e-4
reg.alpha_x = 1.
reg.alpha_y = 1.
reg.alpha_z = 1.
problem.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
mopt = inv.run(m0)
sigopt = mapping*mopt
np.save("sigest", sigopt)
