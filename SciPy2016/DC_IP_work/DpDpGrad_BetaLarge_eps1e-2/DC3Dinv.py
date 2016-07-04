from SimPEG import Mesh, Utils, Maps, Survey, np
from SimPEG import DataMisfit, Regularization, Optimization, Directives, InvProblem, Inversion
from SimPEG.EM.Static import DC, IP
from SimPEG.EM.Static import Utils as StaticUtils
from pymatsolver import MumpsSolver
import timeit
import itertools
import pickle

import sys
sys.path.append("../utilcodes/")
from vizutils import gettopoCC


# Mesh
csx, csy, csz = 25., 25., 25.
ncx, ncy, ncz = 48, 48, 20
npad = 7
hx = [(csx,npad, -1.3),(csx,ncx),(csx,npad, 1.3)]
hy = [(csy,npad, -1.3),(csy,ncy),(csy,npad, 1.3)]
hz = [(csz,npad, -1.3),(csz,ncz), (csz/2.,6)]
mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCN")

# Load model
sigma = mesh.readModelUBC("../VTKout_DC.dat")
airind = sigma == 1e-8

# Identify air cells
airind = sigma==1e-8
mesh2D, topoCC = gettopoCC(mesh, airind)

dx = 25.
x0_core, y0_core, z0_core = -600, -600, -500.

# Define electrode locations

# elecSpace = 100.
# coreOffset = 100 + dx/2.

# Insure that electrode x and y locations fall at cell centres
nskip = 4
elecX = mesh.vectorCCx[np.logical_and(mesh.vectorCCx > -300, mesh.vectorCCx < 320)][::nskip]
elecY = mesh.vectorCCy[np.logical_and(mesh.vectorCCy > -300, mesh.vectorCCy < 320)][::nskip]
nElecX = elecX.size
nElecY = elecY.size

elecX_grid,elecY_grid = np.meshgrid(elecX,elecY)

EW_Lines_Locs = []
EW_Lines_Id =[]
for ii in range(0, nElecY):
    EW_Lines_Locs.append(np.vstack([elecX_grid[ii,:], elecY_grid[ii,:]]).T)
    EW_Lines_Id.append(np.arange(nElecX*ii,nElecX*ii + nElecX,1))

# Create full electrode key which maps electrode Ids and locations
elecLoc = np.vstack(EW_Lines_Locs)

# Drape to topography to get z-value
elecCC_Inds = Utils.closestPoints(mesh2D, elecLoc)
elecLoc_topo = np.c_[elecLoc[:,0], elecLoc[:,1], topoCC[elecCC_Inds]]

nElec = elecLoc_topo.shape[0]
elecId = np.arange(0,nElec,1)
elecLocKey = np.hstack([Utils.mkvc(elecId,2),elecLoc_topo])

# Create dipole-dipole sequence for each EW line
# Identify Tx dipoles on each line
EW_Line_TxElecInd = []
EW_LineId = []
EW_nLines = nElecY
for nr, Line_ElecIds in enumerate(EW_Lines_Id):
    Line_nElec = Line_ElecIds.shape[0]
    for ii in Line_ElecIds[0:-1]:
        for jj in np.arange(ii+1,np.max(Line_ElecIds)+1,1):
            EW_Line_TxElecInd.append([ii , jj])
            EW_LineId.append([nr])

EW_Line_TxElecInd = np.array(EW_Line_TxElecInd)
EW_LineId = np.array(EW_LineId)
nTx = np.array(EW_Line_TxElecInd).shape[0]
nTx_Line = nTx/EW_nLines


# Create data dictionary
dataDict = {}
# nRxList = []
# Iterate over Tx and select possible Rx for each
# nTx = 7 # just for testing
for nr, Tx in enumerate(EW_Line_TxElecInd):
    LineId = EW_LineId[nr]
    useableRxElecs = np.setdiff1d(EW_Lines_Id[LineId[0]],Tx)
    RxPairs = itertools.combinations(useableRxElecs,2) # n choose k combinations

    # Extract data from combinations object
    RxPairList = []
    for ii in RxPairs:
        RxPairList.append(tuple(ii))

    RxPairArray = np.array(RxPairList, dtype=int)
    nRx = RxPairArray.shape[0]
#     nRxList.append([nRx])

    A = Tx[0]*np.ones((nRx,1), dtype=int)
    B = Tx[1]*np.ones((nRx,1), dtype=int)
    LineIdVec = LineId*np.ones((nRx,1), dtype=int)
    dataArray = np.hstack([LineIdVec,A,B,RxPairArray])
    dataDict[nr] = dataArray


srcLists = []
for itx in range (nTx):
    ALoc = elecLocKey[:,1:][dataDict[itx][0,1],:]
    BLoc = elecLocKey[:,1:][dataDict[itx][0,2],:]
    MLoc = elecLocKey[:,1:][dataDict[itx][:,3],:]
    NLoc = elecLocKey[:,1:][dataDict[itx][:,4],:]
    rx = DC.Rx.Dipole(MLoc, NLoc)
    src = DC.Src.Dipole([rx], np.array(ALoc),np.array(BLoc))
    srcLists.append(src)

# Add Gradient Tx and Rxs
# Tx Locations
Aloc_grad = np.array([-600., 0, 0.])
Bloc_grad = np.array([600., 0, 0.])

# Rx Locations
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

rx_x_grad = DC.Rx.Dipole(Mx_dr, Nx_dr)
rx_y_grad = DC.Rx.Dipole(My_dr, Ny_dr)

src_grad = DC.Src.Dipole([rx_x_grad,rx_y_grad], Aloc_grad, Bloc_grad)
srcLists.append(src_grad)

# Model mappings
expmap = Maps.ExpMap(mesh)
actmap = Maps.InjectActiveCells(mesh, ~airind, np.log(1e-8))
mapping = expmap*actmap

survey = DC.Survey(srcLists)
problem = DC.Problem3D_CC(mesh, mapping=mapping)
problem.Solver = MumpsSolver
problem.pair(survey)
mtrue = np.log(sigma)[~airind]
dobs = survey.dpred(mtrue)

# reference model
m0 = np.ones_like(sigma)[~airind]*np.log(1e-4)

regmap = Maps.IdentityMap(nP=m0.size)
std = 0.05
eps = 1e-2
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
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10**1.5)
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
