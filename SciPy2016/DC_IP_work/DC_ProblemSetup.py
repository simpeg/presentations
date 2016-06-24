from SimPEG import Mesh, Utils, Maps, Survey
from SimPEG.EM.Static import DC, IP
from SimPEG.EM.Static import Utils as StaticUtils
from pymatsolver import MumpsSolver

import itertools
%pylab inline


def run(plotIt=True):
    """
        Static: DC: Problem Setup
        =======================

        Design mesh, setup electrode sequence, forward model

    """

    # Create tensor mesh
    # Minimum cell size in each direction
    dx = 10.
    dy = 10.
    dz = 10.

    # Number of core cells in each direction
    nCoreX = 110.
    nCoreY = 110.
    nCoreZ = 60.

    nPadX = 7
    nPadY = 7
    nPadZ = 7

    # Cell widths
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
    topoShift = 200.

    x0_core = -(nCoreX/2)*dx - dx/2.
    y0_core = -(nCoreY/2)*dy - dy/2.
    z0_core = -nCoreZ*dz + topoShift - dz/2.

    # Mesh origin (Bottom SW corner)
    x0 = x0_core - xPadDist
    y0 = y0_core - yPadDist
    z0 = z0_core - zPadDist

    mesh = Mesh.TensorMesh([hx, hy, hz],[x0,y0,z0])
    # # Georeference for topography drape
    # x0 = mesh.x0
    # xc = 300+5.57e5
    # yc = 600+7.133e6
    # zc = 425.
    # mesh._x0 = np.r_[x0[0]+xc, x0[1]+yc, x0[1]+zc]
    # mesh.writeUBC("./Geological_model/TKC_Synth_DC_10mCells.msh")

    print mesh
    print mesh.nC

    mesh.plotSlice(np.ones(mesh.nC)*np.nan, grid=True)


    # Load TKC synthetic model
    sigma = mesh.readModelUBC("./Geological_model/VTKout_DC_10mCells.dat")

    # Plot model sections to check things are reasonable
    mesh.plotSlice(np.log10(sigma), grid=True)
    print mesh.vectorCCz[32]

    mesh.plotSlice(np.log10(sigma), grid=True, normal="Y")


    # Identify air cells
    airind = sigma==1e-8
    mesh2D, topoCC = gettopoCC(mesh, airind)


    # Define electrode locations

    elecSpace = 50.
    coreOffset = 100 + dx/2.

    # x locations
    elecX = np.linspace(x0_core + coreOffset , np.abs(x0_core) - coreOffset, num=((2*(np.abs(x0_core)-coreOffset))/elecSpace) + 1)
    nElecX = elecX.size
    # y locations
    elecY = np.linspace(y0_core + coreOffset , np.abs(y0_core) - coreOffset, num=((2*(np.abs(y0_core)-coreOffset))/elecSpace) + 1)
    nElecY = elecY.size

    # Insure that electrode x and y locations fall at cell centres
    xCC = mesh.vectorCCx[np.logical_and(mesh.vectorCCx > x0_core, mesh.vectorCCx < np.abs(x0_core))]
    yCC = mesh.vectorCCy[np.logical_and(mesh.vectorCCy > y0_core, mesh.vectorCCy < np.abs(y0_core))]

    elecX_unique_sorted, elecX_ind = np.unique(elecX, return_index=True)
    elecX_in_xCC_bool = np.in1d(elecX_unique_sorted, xCC, assume_unique=True)
    print elecX_in_xCC_bool

    # print xCC

    # print elecX

    elecY_unique_sorted, elecY_ind = np.unique(elecY, return_index=True)
    elecY_in_yCC_bool = np.in1d(elecY_unique_sorted, xCC, assume_unique=True)
    print elecY_in_yCC_bool


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

    # print EW_Lines_Id[18]

    NS_Lines_Locs = []
    NS_Lines_Id = []
    for ii in range(0, nElecX):
        NS_Lines_Locs.append(np.vstack([elecX_grid[:,ii], elecY_grid[:,ii]]).T)
        NS_Lines_Id.append(np.arange(ii,nElec,nElecX))

    # print NS_Lines_Id[18]

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
    nTx = 10 # just for testing
    for nr, Tx in enumerate(EW_Line_TxElecInd[0:10,:]):
    #     print nr, Tx
        LineId = EW_LineId[nr]
        useableRxElecs = np.setdiff1d(EW_Lines_Id[LineId[0]],Tx)
        RxPairs = itertools.combinations(useableRxElecs,2) # n choose k combinations

        # Extract data from combinations object
        RxPairList = []
        for ii in RxPairs:
            RxPairList.append(tuple(ii))

        RxPairArray = np.array(RxPairList)
        nRx = RxPairArray.shape[0]
    #     nRxList.append([nRx])

        A = Tx[0]*np.ones((nRx,1))
        B = Tx[1]*np.ones((nRx,1))
        LineIdVec = LineId*np.ones((nRx,1))
        dataArray = np.hstack([LineIdVec,A,B,RxPairArray])
        dataDict[nr] = dataArray


    # Create Tx and Rx data objects for the survey object

    FullElecSeqList = []
    for Tx in dataDict.keys():
        FullElecSeqList.append(dataDict[Tx][:,1:5])

    # print "nTx = %i" % (nTx)

    FullElecSeqArray = np.array(FullElecSeqList)
    print FullElecSeqArray.shape
    print nTx
    FullElecSeq = FullElecSeqArray.reshape(nTx*nRx,4)
    FullElecSeq.shape

    nData = FullElecSeq.shape[0]
    # print nData

    AIds = FullElecSeq[:,0]
    BIds = FullElecSeq[:,1]
    MIds = FullElecSeq[:,2]
    NIds = FullElecSeq[:,3]

    ALoc = np.zeros([nData,3])
    BLoc = np.zeros([nData,3])
    MLoc = np.zeros([nData,3])
    NLoc = np.zeros([nData,3])


    print elecLoc_topo.shape

    for ii in range(0, nElec-1):
        AInd = np.where(AIds == ii)
        ALoc[AInd,:] = elecLoc_topo[ii,:]

        BInd = np.where(BIds == ii)
        BLoc[BInd,:] = elecLoc_topo[i,:]

        MInd = np.where(MIds == ii)
        MLoc[MInd,:] = elecLoc_topo[i,:]

        NInd = np.where(NIds == ii)
        NLoc[NInd,:] = elecLoc_topo[ii,:]

    RxData = DC.Rx.Dipole(MLoc, NLoc)
    TxData = DC.Src.Dipole([RxData], np.array(ALoc),np.array(BLoc))

    # Model mappings
    expmap = Maps.ExpMap(mesh)
    actmap = Maps.InjectActiveCells(mesh, ~airind, np.log(1e-8))
    mapping = expmap*actmap

    # reference model
    m0 = np.ones_like(sigma)[~airind]*np.log(1e-4)


    # Setup forward modelling
    survey = DC.Survey([TxData])
    problem = DC.Problem3D_CC(mesh, mapping=mapping)
    problem.Solver = MumpsSolver
    problem.pair(survey)
    mtrue = np.log(sigma)[~airind]
    f = problem.fields(mtrue)
    dobs = survey.dpred(mtrue, f=f)
    survey.dobs = dobs


    # if plotIt:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1,1, figsize = (3, 6))
    #     plt.semilogx(sigma[active], mesh.vectorCCz[active])
    #     ax.set_ylim(-600, 0)
    #     ax.set_xlim(1e-4, 1e-2)
    #     ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
    #     ax.set_ylabel('Depth (m)', fontsize = 14)
    #     ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)


    # rxOffset=1e-3
    # rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 30]]), np.logspace(-5,-3, 31), 'bz')
    # src = EM.TDEM.SrcTDEM_VMD_MVP([rx], np.array([0., 0., 80]))
    # survey = EM.TDEM.SurveyTDEM([src])
    # prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)

    # prb.Solver = MumpsSolver
    # prb.timeSteps = [(1e-06, 20),(1e-05, 20), (0.0001, 20)]
    # prb.pair(survey)

    # # create observed data
    # std = 0.05

    # survey.dobs = survey.makeSyntheticData(mtrue,std)
    # survey.std = std
    # survey.eps = 1e-5*np.linalg.norm(survey.dobs)

    # if plotIt:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1,1, figsize = (10, 6))
    #     ax.loglog(rx.times, survey.dtrue, 'b.-')
    #     ax.loglog(rx.times, survey.dobs, 'r.-')
    #     ax.legend(('Noisefree', '$d^{obs}$'), fontsize = 16)
    #     ax.set_xlabel('Time (s)', fontsize = 14)
    #     ax.set_ylabel('$B_z$ (T)', fontsize = 16)
    #     ax.set_xlabel('Time (s)', fontsize = 14)
    #     ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)

    # dmisfit = DataMisfit.l2_DataMisfit(survey)
    # regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
    # reg = Regularization.Tikhonov(regMesh)
    # opt = Optimization.InexactGaussNewton(maxIter = 5)
    # invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # # Create an inversion object
    # beta = Directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    # betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
    # inv = Inversion.BaseInversion(invProb, directiveList=[beta,betaest])
    # m0 = np.log(np.ones(mtrue.size)*sig_half)
    # reg.alpha_s = 1e-2
    # reg.alpha_x = 1.
    # prb.counter = opt.counter = Utils.Counter()
    # opt.LSshorten = 0.5
    # opt.remember('xc')

    # mopt = inv.run(m0)

    # if plotIt:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1,1, figsize = (3, 6))
    #     plt.semilogx(sigma[active], mesh.vectorCCz[active])
    #     plt.semilogx(np.exp(mopt), mesh.vectorCCz[active])
    #     ax.set_ylim(-600, 0)
    #     ax.set_xlim(1e-4, 1e-2)
    #     ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
    #     ax.set_ylabel('Depth (m)', fontsize = 14)
    #     ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    #     plt.legend(['$\sigma_{true}$', '$\sigma_{pred}$'])
    #     plt.show()


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

