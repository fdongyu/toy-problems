# 2D Shallow Water Equations

Please install PETSc version v3.17.3 (Github hash [c5e9cb](https://github.com/petsc/petsc/commit/c5e9cb188e4dab0f70a8981c0ddfcef5a478e87f)) before running the example.

ex1.c, ex1f.F90, ex2a.c, ex2b.c use Roe's approximate Reimann solver for the 2D Shallow Water Equations.

## 1. ex1.c
ex1.c solves the followling equations for the Dam Break on a flat surface.
$$
\begin{equation}
  \frac{\partial h}{\partial t} + \frac{\partial uh}{\partial x} + \frac{\partial vh}{\partial y} = 0
\end{equation}
$$
$$
\begin{equation}
  \frac{\partial uh}{\partial t} + \frac{\partial \left(u^{2}h+\frac{1}{2}gh^{2}\right)}{\partial x} + \frac{\partial uvh}{\partial y} = 0
\end{equation}
$$
$$
\begin{equation}
  \frac{\partial vh}{\partial t} + \frac{\partial uvh}{\partial x}+ \frac{\partial \left(v^{2}h+\frac{1}{2}gh^{2}\right)}{\partial y}  = 0 
\end{equation}
$$
* Using ``DMDA`` for solving Dam Break example on a structured mesh. 
* User can use ``-Nx``, ``-Ny`` to specify the number of cells in X, and Y direction. Use ``-Nt`` to specify total simulation steps, and ``-dt`` to specify time step in sec. 
* Use ``-save`` to save the outputs, and use the ``show_ex1.m`` show vizulize the outputs.

```
make ex1
mpiexec -n2 ./ex1 -Nt 100 -dt 0.1 -Nx 100 -Ny 100 -save
```

## 2. ex2 (TODO)