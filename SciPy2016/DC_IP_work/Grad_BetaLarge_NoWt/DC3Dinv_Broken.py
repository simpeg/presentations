from SimPEG import Mesh, Utils, Maps, Survey, np
from SimPEG import DataMisfit, Regularization, Optimization, Directives, InvProblem, Inversion
from SimPEG.EM.Static import DC, IP
from SimPEG.EM.Static import Utils as StaticUtils
from pymatsolver import MumpsSolver
import timeit
import itertools
import pickle

csx, csy, csz = 25., 25., 25.
ncx, ncy, ncz = 48, 48, 20
npad = 7
hx = [(csx,npad, -1.3),(csx,ncx),(csx,npad, 1.3)]
hy = [(csy,npad, -1.3),(csy,ncy),(csy,npad, 1.3)]
hz = [(csz,npad, -1.3),(csz,ncz), (csz/2.,6)]
mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCN")

fname = "DCsetup_Grad"
DCsetup = pickle.load(open(fname,"rb"))

sigma = DCsetup["sigma"]
airind = sigma == 1e-8
# nTx = len(DCsetup["Tx"])

ALoc, BLoc = DCsetup["Tx"][0], DCsetup["Tx"][1]
MLoc, NLoc = DCsetup["Rx"][:][0], DCsetup["Rx"][:][1]
rx = DC.Rx.Dipole(MLoc, NLoc)
src = DC.Src.Dipole([rx], np.array(ALoc),np.array(BLoc))

expmap = Maps.ExpMap(mesh)
actmap = Maps.InjectActiveCells(mesh, ~airind, np.log(1e-8))
mapping = expmap*actmap

survey = DC.Survey([src])
problem = DC.Problem3D_CC(mesh, mapping=mapping)
problem.Solver = MumpsSolver
problem.pair(survey)
mtrue = np.log(sigma)[~airind]
dobs = survey.dpred(mtrue)

# reference model
m0 = np.ones_like(sigma)[~airind]*np.log(1e-4)

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
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
save = Directives.SaveOutputEveryIteration()
savemodel = Directives.SaveModelEveryIteration()
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, save, target, savemodel])
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
