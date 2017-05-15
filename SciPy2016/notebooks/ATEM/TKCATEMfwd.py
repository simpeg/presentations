from SimPEG import EM, Mesh, Utils
import numpy as np
from scipy.constants import mu_0
from pymatsolver import PardisoSolver
import cPickle as pickle

TKCATEMexample = pickle.load( open( "../TKCATEMexample.p", "rb" ) )

mesh = TKCATEMexample["mesh"]
sigma = TKCATEMexample["sigma"]
xyz = TKCATEMexample["xyz"]
ntx = xyz.shape[0]

# TDEM Survey
srcLists = []
times = np.logspace(-4, np.log10(2e-3), 10)
for itx in range(ntx):
    rx = EM.TDEM.Rx(xyz[itx,:].reshape([1,-1]), times, 'bz')
    src = EM.TDEM.Src.CircularLoop([rx], waveform=EM.TDEM.Src.StepOffWaveform(), loc=xyz[itx, :].reshape([1, -1]), radius = 13.) # same src location as FDEM problem
    srcLists.append(src)

# TDEM Problem
survey = EM.TDEM.Survey(srcLists)
problem = EM.TDEM.Problem_b(mesh, verbose=True)
timeSteps = [(1e-5, 5), (1e-4, 10), (5e-4, 10)]
problem.timeSteps = timeSteps
problem.pair(survey)
problem.Solver = MumpsSolver

# Simulate
dpred = survey.dpred(sigma)
TKCATEMexample = {"mesh": mesh, "sigma":sigma, "xyz":xyz, "ntx":ntx,
                  "times":times, "timeSteps":problem.timeSteps, "dpred":dpred}
pickle.dump( TKCATEMexample, open( "TKCATEMexample_fwd.p", "wb" ) )
