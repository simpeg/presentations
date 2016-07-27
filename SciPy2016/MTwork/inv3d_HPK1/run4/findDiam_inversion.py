# Import modules
import numpy as np, sys, os, time, gzip, cPickle as pickle, scipy
sys.path.append('/tera_raid/gudni/gitCodes/simpeg')
sys.path.append('/tera_raid/gudni')
from pymatsolver import MumpsSolver
import SimPEG as simpeg
from SimPEG import NSEM
from numpy.lib import recfunctions as rFunc

# Function to get data and data info
def getDataInfo(MTdata):

    dL, freqL, rxTL = [], [], []

    for src in MTdata.survey.srcList:
        for rx in src.rxList:
            dL.append(MTdata[src,rx])
            freqL.append(np.ones(rx.nD)*src.freq)
            rxTL.extend( ((rx.rxType+' ')*rx.nD).split())
    return np.concatenate(dL), np.concatenate(freqL), np.array(rxTL)

# Script to read MT data and run an inversion.

# Load the data
drecAll = np.load('../../MTdataStArr_nsmesh_HKPK1.npy')
# Select larger frequency band for the MT data
indMTFreq = np.sum( [drecAll['freq'] == val for val in  np.unique(drecAll['freq'])[5::]] ,axis=0,dtype=bool)
mtRecArr = drecAll[indMTFreq][['freq','x','y','z','tzx','tzy']]
dUse = NSEM.Data.fromRecArray(mtRecArr)

# Extract to survey
survey = dUse.survey

# # Add noise to the data
dobs, freqArr, rxT = getDataInfo(dUse)
# Set the data

survey.dobs = dobs
# Assign std based on- and off-diagonal parts of the impedance tensor
std = np.ones_like(dobs)*.025 # 5% on all off-diagonals
# std[np.array([ ('xx' in l or 'yy' in l) for l in rxT])] = 0.15 # 15% on the on-diagonal
survey.std = np.abs(survey.dobs*std) #+ 0.01*np.linalg.norm(survey.dobs) #survey.dobs*0 + std
# Estimate a floor for the data.
# Use the 10% of the mean of the off-diagonals for each frequency
floor = np.zeros_like(dobs)
offind = np.array([('zxy' in l or 'zyx' in l) for l in rxT],bool)
onind = np.array([('zxx' in l or 'zyy' in l) for l in rxT],bool)
tipind = np.array([('tzx' in l or 'tzy' in l) for l in rxT],bool)

for f in np.unique(freqArr):
    freqInd = freqArr == f
    floorFreq = floor[freqInd]
    offD = np.sort(np.abs(dobs[freqInd*offind]))
    floor[freqInd] = 0.0001*np.mean(offD)
    onD = np.sort(np.abs(dobs[freqInd*onind]))
    floor[freqInd*onind] = 0.1*np.mean(onD)
    # Constant floor for the tipper.
    floor[freqInd*tipind] = 0.001

# Assign the data weight
Wd = 1./(survey.std + floor)

# Make the mesh
mesh, modDict = simpeg.Mesh.TensorMesh.readVTK('../../nsmesh_HPVK1_inv.vtr')
sigma = modDict['S/m']

# Make the mapping
active = sigma > 9.999e-7
actMap = simpeg.Maps.ActiveCells(mesh, active, np.log(1e-8), nC=mesh.nC)
mappingExpAct = simpeg.Maps.ExpMap(mesh) * actMap

# sigma1d = 1e-8 * mesh.vectorCCz
# sigma1d[mesh.vectorCCz < 750] = 1e-2
sigmaBG = np.ones_like(sigma)*1e-8
sigmaBG[active] = 1e-4
sigma1d = mesh.r(sigmaBG,'CC','CC','M')[0,0,:]

# Make teh starting model
m_0 = np.log(sigmaBG[active])
## Setup the problem object
problem = NSEM.Problem3D_ePrimSec(mesh,mapping=mappingExpAct,sigmaPrimary = sigma1d)
problem.verbose = True
# Change the solver
problem.Solver = MumpsSolver
problem.pair(survey)

## Setup the inversion proceedure
C =  simpeg.Utils.Counter()

# Set the optimization
# opt = simpeg.Optimization.ProjectedGNCG(maxIter = 50)
# opt.lower = np.log(1e-5) # Set bounds to
# opt.upper = np.log(10)
opt = simpeg.Optimization.InexactGaussNewton(maxIter = 36)
opt.counter = C
opt.LSshorten = 0.5
opt.remember('xc')
# Need to add to the number of iter per beta.
# Data misfit
dmis = simpeg.DataMisfit.l2_DataMisfit(survey)
dmis.Wd = Wd
# Regularization
# regMap = simpeg.Maps.InjectActiveCellsTopo(mesh,active) # valInactive=
reg = simpeg.Regularization.Tikhonov(mesh,indActive=active)
reg.mref = m_0
reg.alpha_s = 1e-7
reg.alpha_x = 1.
reg.alpha_y = 1.
reg.alpha_z = 1.
reg.alpha_xx = 0.
reg.alpha_yy = 0.
reg.alpha_zz = 0.
# Inversion problem
invProb = simpeg.InvProblem.BaseInvProblem(dmis, reg, opt)
invProb.counter = C
# Beta cooling
beta = simpeg.Directives.BetaSchedule()
beta.coolingRate = 3 # Number of beta iterations
beta.coolingFactor = 8.
betaest = simpeg.Directives.BetaEstimate_ByEig(beta0_ratio=1000.)
# invProb.beta = 1e6.
targmis = simpeg.Directives.TargetMisfit()
targmis.target = 0.5 * survey.nD
# saveModel = simpeg.Directives.SaveModelEveryIteration()
saveDict = simpeg.Directives.SaveOutputDictEveryIteration()
# Create an inversion object
inv = simpeg.Inversion.BaseInversion(invProb, directiveList=[beta,targmis,saveDict])

# Print
print 'Target Misfit is: {:.1f}'.format(targmis.target)

# Run the inversion
mopt = inv.run(m_0)