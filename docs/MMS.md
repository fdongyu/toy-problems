

## Generating meshes

Run ``../share/meshes/generate_mesh_MMS.m`` and change dx.

## Run Simulations

```
mpiexec -n 1 ./ex2b_MMS -savef true -output_prefix MMS_dx1 -mesh ../share/meshes/MMS_mesh_dx1.exo -initial_condition ../share/initial_conditions/MMS_dx1.IC -dt 0.01 -ts_max_time 5
```

## Plot solution

Exact solution ``../share/postprocessing/manufactured_solution.py``
Simulation output ``../share/postprocessing/RDycore_solution.py``
