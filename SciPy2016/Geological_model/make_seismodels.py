from collections import OrderedDict
import numpy as np
import h5py
import multiprocessing
from SimPEG import Mesh
from dommagic import gocad2vtk

mshfile = 'seismic.msh'
hdf5file = 'masks.hdf5'
mesh = Mesh.TensorMesh.readUBC(mshfile)

geosurf = OrderedDict(
[ #   Unit       Filename               bcflag  inflag
    ('air',     ('CDED_Lake_Coarse.ts', False,  True )),
    ('till',    ('Till.ts',             True,   True )),
    ('xvk',     ('XVK.ts',              True,   True )),
    ('pk1',     ('PK1.ts',              True,   True )),
    ('pk2',     ('PK2.ts',              True,   True )),
    ('pk3',     ('PK3.ts',              True,   True )),
    ('hk',      ('HK1.ts',              True,   True )),
    ('vk',      ('VK.ts',               True,   True )),
])

def getmask(key):
    mask = np.zeros(mesh.nC, dtype=np.bool)
    indx = gocad2vtk(geosurf[key][0], mesh, geosurf[key][1], geosurf[key][2])
    mask[indx] = True
    return mask

keys = geosurf.keys()
p = multiprocessing.Pool()
masks = list(p.map(getmask, keys))
p.close()
aind = keys.index('air')
masks[aind] = True ^ masks[aind]

mfile = h5py.File(hdf5file)
for i, key in enumerate(keys):
    mfile[key] = masks[i].reshape((mesh.nCz, mesh.nCy, mesh.nCx))
