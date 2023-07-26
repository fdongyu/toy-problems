import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/qfs/people/feng779/RDycore/petsc/lib/petsc/bin')

from PetscBinaryIO import *
from spatial_map import RDycore_spatial_map

import pdb



Lx = 5 # m
Ly = 5 # m

h0 = 0.005
tend = 5 # seconds
#dx = 1 # [0.05, 0.1, 0.25, 0.5, 1]
dx = 1

output = '../../swe/outputs/MMS_dx1_dt_0.010000_final_solution.dat'
data = PetscBinaryIO().readBinaryFile(output)[0]
data = np.reshape(data, (3,int(len(data)/3)), order='F');

h = data[0,:]
uh= data[1,:]
vh = data[2,:]

pdb.set_trace()

#### plot 
mesh_file = '../meshes/MMS_mesh_dx{}.exo'.format(dx)
RSM = RDycore_spatial_map()
RSM.xlim = [0, Lx]
RSM.ylim = [0, Ly]
RSM.vmax = h0
RSM.vmin = 0
RSM.read_mesh(mesh_file)

RSM.vmax = 0.015
RSM.vmin = h0

RSM.figname = 'RDycore_final_dx{}.png'.format(dx)
RSM.spatial_map(h, 'h [m]')
