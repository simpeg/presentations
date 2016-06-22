from SimPEG import Mesh, Utils, Maps, Survey
from SimPEG.EM.Static import DC, IP, Utils
from pymatsolver import MumpsSolver
import numpy as np


def run(plotIt=True):
    """
        EM: TDEM: 1D: Inversion
        =======================

        Here we will create and run a TDEM 1D inversion.

    """

    # Create tensor mesh
	# Minimum cell size in each direction
	dx = 10.
	dy = 10.
	dz = 10.

	# Number of core cells in each direction
	nCoreX = 120.
	nCoreY = 120.
	nCoreZ = 50.

	nPadX = 7
	nPadY = 7
	nPadZ = 7


	# Cell widths
	# hx = [(dx,nCoreX)]
	# hy = [(dy,nCoreY)]
	# hz = [(dz,nCoreZ)]
	hx = [(dx,nPadX, -1.3),(dx,nCoreX),(dx,nPadX, 1.3)]
	hy = [(dy,nPadY, -1.3),(dy,nCoreY),(dy,nPadY, 1.3)]
	hz = [(dz,nPadZ, -1.3),(dz,nCoreZ),(dz,nPadZ, 1.3)]

	# Calculate X padding distance
	hPadX = np.zeros([nPadX+1,1])
	hPadX[0] = dx
	for i in range (1,nPadX+1):
	    hPadX[i] = hPadX[i-1]*1.3

	xPadDist = np.max(np.cumsum(hPadX[1:nPadX+1]))

	# Calculate Y padding distance
	hPadY = np.zeros([nPadY+1,1])
	hPadY[0] = dy
	for i in range (1,nPadY+1):
	    hPadY[i] = hPadY[i-1]*1.3

	yPadDist = np.max(np.cumsum(hPadY[1:nPadY+1]))

	# Calculate Z padding distance
	hPadZ = np.zeros([nPadZ+1,1])
	hPadZ[0] = dz
	for i in range (1,nPadZ+1):
	    hPadZ[i] = hPadZ[i-1]*1.3

	zPadDist = np.max(np.cumsum(hPadZ[1:nPadZ+1]))

	# Desired Core mesh origin (Bottom SW corner)
	x0_core = -(nCoreX/2)*dx - dx/2.
	y0_core = -(nCoreY/2)*dy - dy/2.
	z0_core = -(nCoreZ)*dz - dz/2.

	# Mesh origin (Bottom SW corner)
	x0 = x0_core - xPadDist
	y0 = y0_core - yPadDist
	z0 = z0_core - zPadDist

	mesh = Mesh.TensorMesh([hx, hy, hz],[x0,y0,z0])

    # csx, csy, csz = 10., 10., 10.
    # ncx, ncy, ncz = 120, 120, 50
    # npad = 7
    # hx = [(csx,npad, -1.3),(csx,ncx),(csx,npad, 1.3)]
    # hy = [(csy,npad, -1.3),(csy,ncy),(csy,npad, 1.3)]
    # hz = [(csz,npad, -1.3),(csz,ncz), (csz/2.,6)]
    # mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCN")

    # Load TKC synthetic model
    sigma = mesh.readModelUBC("VTKout_DC.dat")

    airind = sigma==1e-8
    mesh2D, topoCC = gettopoCC(mesh, airind)

    # Define electrode locations

    # Find start and end locations of each line
    lineSpace = 50.
    coreOffset = 150.
	lineY = np.linspace(y0_core + coreOffset , np.abs(y0_core) - coreOffset, num=((2*(np.abs(y0_core)-coreOffset))/lineSpace))


    Utils.gen_DCIPsurvey()

    # active = mesh.vectorCCz<0.
    # layer = (mesh.vectorCCz<0.) & (mesh.vectorCCz>=-100.)
    # actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    # mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
    # sig_half = 2e-3
    # sig_air = 1e-8
    # sig_layer = 1e-3
    # sigma = np.ones(mesh.nCz)*sig_air
    # sigma[active] = sig_half
    # sigma[layer] = sig_layer
    # mtrue = np.log(sigma[active])


    # if plotIt:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1,1, figsize = (3, 6))
    #     plt.semilogx(sigma[active], mesh.vectorCCz[active])
    #     ax.set_ylim(-600, 0)
    #     ax.set_xlim(1e-4, 1e-2)
    #     ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
    #     ax.set_ylabel('Depth (m)', fontsize = 14)
    #     ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)


    rxOffset=1e-3
    rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 30]]), np.logspace(-5,-3, 31), 'bz')
    src = EM.TDEM.SrcTDEM_VMD_MVP([rx], np.array([0., 0., 80]))
    survey = EM.TDEM.SurveyTDEM([src])
    prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)

    prb.Solver = MumpsSolver
    prb.timeSteps = [(1e-06, 20),(1e-05, 20), (0.0001, 20)]
    prb.pair(survey)

    # create observed data
    std = 0.05

    survey.dobs = survey.makeSyntheticData(mtrue,std)
    survey.std = std
    survey.eps = 1e-5*np.linalg.norm(survey.dobs)

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize = (10, 6))
        ax.loglog(rx.times, survey.dtrue, 'b.-')
        ax.loglog(rx.times, survey.dobs, 'r.-')
        ax.legend(('Noisefree', '$d^{obs}$'), fontsize = 16)
        ax.set_xlabel('Time (s)', fontsize = 14)
        ax.set_ylabel('$B_z$ (T)', fontsize = 16)
        ax.set_xlabel('Time (s)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

    dmisfit = DataMisfit.l2_DataMisfit(survey)
    regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    reg = Regularization.Tikhonov(regMesh)
    opt = Optimization.InexactGaussNewton(maxIter = 5)
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Create an inversion object
    beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    inv = Inversion.BaseInversion(invProb, directiveList=[beta,betaest])
    m0 = np.log(np.ones(mtrue.size)*sig_half)
    reg.alpha_s = 1e-2
    reg.alpha_x = 1.
    prb.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')

    mopt = inv.run(m0)

    if plotIt:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1, figsize = (3, 6))
        plt.semilogx(sigma[active], mesh.vectorCCz[active])
        plt.semilogx(np.exp(mopt), mesh.vectorCCz[active])
        ax.set_ylim(-600, 0)
        ax.set_xlim(1e-4, 1e-2)
        ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
        ax.set_ylabel('Depth (m)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
        plt.legend(['$\sigma_{true}$', '$\sigma_{pred}$'])
        plt.show()


if __name__ == '__main__':
    run()


def gettopoCC(mesh, airind):
# def gettopoCC(mesh, airind):
    """
        Get topography from active indices of mesh.
    """
    mesh2D = Mesh.TensorMesh([mesh.hx, mesh.hy], mesh.x0[:2])
    zc = mesh.gridCC[:,2]
    AIRIND = airind.reshape((mesh.vnC[0]*mesh.vnC[1],mesh.vnC[2]), order='F')
    ZC = zc.reshape((mesh.vnC[0]*mesh.vnC[1], mesh.vnC[2]), order='F')
    topo = np.zeros(ZC.shape[0])
    topoCC = np.zeros(ZC.shape[0])
    for i in range(ZC.shape[0]):
        ind  = np.argmax(ZC[i,:][~AIRIND[i,:]])
        topo[i] = ZC[i,:][~AIRIND[i,:]].max() + mesh.hz[~AIRIND[i,:]][ind]*0.5
        topoCC[i] = ZC[i,:][~AIRIND[i,:]].max()
    XY = Utils.ndgrid(mesh.vectorCCx, mesh.vectorCCy)
    return mesh2D, topoCC

