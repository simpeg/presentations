from SimPEG import Mesh, Utils
import numpy as np

import matplotlib.pyplot as plt

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


def viz(mesh, sigma, ind, airind, normal="Z", ax=None, label="Conductivity (S/m)", scale="log", clim=(-4, -1)):
    if normal == "Z":
        fig = plt.figure(figsize=(5*1.2, 5))
        ax = plt.subplot(111)
    elif normal == "Y":
        fig = plt.figure(figsize=(5*1.2, 2.5))
        ax = plt.subplot(111)
    temp = sigma.copy()

    if scale == "log":
        temp = np.log10(temp)

    temp[airind] = np.nan

    dat = mesh.plotSlice(temp, ind=ind, clim=clim, normal=normal, grid=True, pcolorOpts={"cmap":"viridis"}, ax=ax)
    if normal == "Z":
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500.)
        ax.set_title(("Eleveation at %.1f m")%(mesh.vectorCCy[ind]))
    elif normal == "Y":
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 0.)
        ax.set_title(("Northing at %.1f m")%(mesh.vectorCCy[ind]))
    if scale == "log":
        cbformat = "$10^{%1.1f}$"
    elif scale == "linear":
        cbformat = "%.1e"

    cb = plt.colorbar(dat[0], format=cbformat, ticks=np.linspace(clim[0], clim[1], 3))
    cb.set_label(label)
    # plt.show()
    return ax

def vizEJ(mesh, sigma, ind, f, src, airind, normal="Z", ftype="E", clim=None):
    if normal == "Z":
        fig = plt.figure(figsize=(5*1.2, 5))
        ax = plt.subplot(111)
    else:
        fig = plt.figure(figsize=(5*1.2, 2.5))
        ax = plt.subplot(111)
    temp = sigma.copy()
    temp[airind] = np.nan

    if ftype == "E":
        dat=mesh.plotSlice(f[src,'e'], vType="F", view="vec", ind=ind, normal=normal, grid=False, streamOpts={'color':'w'}, pcolorOpts={"cmap":"viridis"}, ax=ax)
        ax.set_title("Electric fields (V/m)")
    elif ftype == "charg":
        dat=mesh.plotSlice(f[src,'charge'], ind=ind, normal=normal, pcolorOpts={"cmap":"viridis"}, ax=ax)
        ax.set_title("Electric charges (C)")
    elif ftype == "J":
        dat=mesh.plotSlice(f[src,'j'], vType="F", view="vec", ind=ind, normal=normal, grid=False, streamOpts={'color':'w'}, pcolorOpts={"cmap":"viridis"}, ax=ax)
        ax.set_title("Electric currents (V/m)")
    ax.set_xlabel("Easting (m)")

    vmin, vmax = dat[0].get_clim()
    if normal == "Z":
        ax.set_xlim(-700, 700)
        ax.set_ylim(-700, 700.)
        ax.set_ylabel("Northing (m)")
    else:
        ax.set_xlim(-700, 700)
        ax.set_ylim(-700, 0.)
        ax.set_ylabel("Elevation (m)")
    cb = plt.colorbar(dat[0], format="%1.1e", ticks=np.linspace(vmin, vmax, 3))
