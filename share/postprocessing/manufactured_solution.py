import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/qfs/people/feng779/RDycore/petsc/lib/petsc/bin')

from spatial_map import RDycore_spatial_map

import pdb


"""
Method of Manufactured Solutions
h(x,y,t) = h0 ( 1+sin(pi x/Lx)sin(pi y/Ly) ) exp(t/t0)
u(x,y,t) = u0 ( 1+sin(pi x/Lx)sin(pi y/Ly) ) exp(t/t0)
v(x,y,t) = v0 ( 1+sin(pi x/Lx)sin(pi y/Ly) ) exp(t/t0)
"""

Lx = 5 # m
Ly = 5 # m

h0 = 0.005 
u0 = 0.025
v0 = 0.025
t0 = 20

tend = 5 # seconds

#### exact solution is h(x,y,t=5), u(x,y,t=5), v(x,y,t=5)

def h_solution(x,y,t):
    return h0* ( 1+np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly) ) * np.exp(t/t0)

def u_solution(x,y,t):
    return u0* ( 1+np.sin(np.pi*x/Lx)*np.sin(np.pi*y/Ly) ) * np.exp(t/t0)

"""
# structured grid 
dx = 1 # [0.05, 0.1, 0.25, 0.5, 1]
dy = dx
dt = 0.01

tt = np.arange(0, tend+dt, dt)
xx = np.arange(0, Lx+dx, dx) + dx/2
yy = np.arange(0, Ly+dy, dy) + dy/2

Nt = tt.shape[0]
Nx = xx.shape[0]
Ny = yy.shape[0]

xx, yy = np.meshgrid(xx, yy)

h_exact_initial = h_solution(xx, yy, tt[0])
h_exact_end = h_solution(xx, yy, tt[-1])
"""

dx = 1 # [0.05, 0.1, 0.25, 0.5, 1]
dy = dx
dt = 0.01
tt = np.arange(0, tend+dt, dt)


#### plot 
mesh_file = '../meshes/MMS_mesh_dx{}.exo'.format(dx)
RSM = RDycore_spatial_map()
RSM.xlim = [0, Lx]
RSM.ylim = [0, Ly]
RSM.vmax = h0
RSM.vmin = 0
RSM.read_mesh(mesh_file)

xc = np.mean(RSM.trix, axis=1)
yc = np.mean(RSM.triy, axis=1)

h_exact_initial = h_solution(xc, yc, tt[0])
h_exact_final = h_solution(xc, yc, tt[-1])


RSM.vmax = 0.015
RSM.vmin = h0

#RSM.figname = 'figures/initial_dx{}.png'.format(dx)
#RSM.spatial_map(h_exact_initial, 'h [m]')
RSM.figname = 'exact_final_dx{}.png'.format(dx)
RSM.spatial_map(h_exact_final, 'h [m]')
