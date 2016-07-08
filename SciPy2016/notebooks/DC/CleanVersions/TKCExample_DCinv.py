from SimPEG import DataMisfit, Regularization, Optimization, Directives, InvProblem, Inversion
import pickle

# Run survey setup and forward modelling
run TKCExample_DCfwd.py


# Depth weighting
depth = 1./(abs(mesh.gridCC[:,2]-zc))**1.5
depth = depth/depth.max()

# Setup inversion object
regmap = Maps.IdentityMap(nP=m0.size)
# Assign uncertainties
std = 0.05
eps = 1e-3
survey.std = std
survey.eps = eps
survey.dobs = dobs
# Define datamisfit portion of objective function
dmisfit = DataMisfit.l2_DataMisfit(survey)
# Define regulatization (model objective function)
reg = Regularization.Simple(mesh, mapping=regmap, indActive=~airind)
reg.wght = depth[~airind]
opt = Optimization.InexactGaussNewton(maxIter = 20)
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
# Define inversion parameters
beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
save = Directives.SaveOutputEveryIteration()
savemodel = Directives.SaveModelEveryIteration()
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta,betaest, save, target, savemodel])
reg.alpha_s = 1e-1
reg.alpha_x = 1.
reg.alpha_y = 1.
reg.alpha_z = 1.
problem.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')

# Run inversion
mopt = inv.run(m0)

# Apply mapping to model to get and save recovered conductivity
sigopt = mapping*mopt
np.save("sigest_singlesrc_withdweights", sigopt)

# Calculate dpred
dpred = survey.dpred(np.log(sigopt_withdepth[~airind]))

# Pickle results for easy access
Results = {"model_true":sigma, "model_pred":sigopt, "Obs":dobs, "Pred":dpred}
outputs = open("DCresults", 'wb')
pickle.dump(Results, outputs)
outputs.close()


