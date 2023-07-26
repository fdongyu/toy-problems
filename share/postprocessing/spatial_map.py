import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
import sys
sys.path.append('/qfs/people/feng779/RDycore/petsc/lib/petsc/bin')

from PetscBinaryIO import *

import pdb


class RDycore_spatial_map(object):
    """
    class for plotting a spatial map
    Input: meshfile, outputfile
    """

    figname = None
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    cb_on = True
    xlim = None
    ylim = None
    vmax = None
    vmin = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def read_mesh(self, mesh_file):
        """
        mesh = '/qfs/people/feng779/RDycore/toy-problems/share/meshes/DamBreak_grid5x10.exo'
        """

        nc = Dataset(mesh_file)
        #print (nc)
        tri = nc.variables['connect1'][:].data
        self.coordx = nc.variables['coordx'][:].data
        self.coordy = nc.variables['coordy'][:].data
        self.trix = self.coordx[tri-1]
        self.triy = self.coordy[tri-1]
        nc.close()

        self.cellxy = []
        for i in range(self.trix.shape[0]):
            verts = np.vstack([self.trix[i,:], self.triy[i,:]]).T
            self.cellxy.append(verts)

    def spatial_map(self, var, varname):
        """
        var: 1D array of the variable matching the dimension of the input mesh
        """

        """
        output = '/qfs/people/feng779/RDycore/toy-problems/swe/outputs/dambreak5x10_dt_0.100000_final_solution.dat'
        data = PetscBinaryIO().readBinaryFile(output)[0]
data = np.reshape(data, (3,int(len(data)/3)), order='F');
        h = data[0,:]
        uh= data[1,:]
        vh = data[2,:]

        vmin = 5
        vmax = 10
        var = h
        varname = 'h [m]'
        """
        if self.vmin == None or self.vmax == None:
            self.vmin = self.vmin
            self.vmax = self.vmax

        if self.xlim == None or self.ylim == None:
            self.xlim = [self.coordx.min(), self.coordx.max()]
            self.ylim = [self.coordy.min(), self.coordy.max()]
        

        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        collection = PolyCollection(self.cellxy, cmap=self.cmap, closed=True)
        collection.set_array(var[:])
        collection.set_edgecolors('k')
        #collection.set_edgecolors(collection.to_rgba(var[:])) 
        #collection.set_linewidths(0.005)
        collection.set_linewidths(0.1)
        collection.set_clim(vmin=self.vmin, vmax=self.vmax)
        cs = ax.add_collection(collection)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim);

        if self.cb_on: # colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cb = fig.colorbar(cs, cax=cax, orientation='vertical')
            cb.ax.tick_params(labelsize=12)
            cb.ax.yaxis.offsetText.set_fontsize(12)
            cb.set_label(varname, fontsize=14)

        ax.set_aspect('equal')
        fig.tight_layout()
        if self.figname != None:
            plt.savefig(self.figname)
            plt.close()
        else:
            raise IOError("Specify a path for the figure")
