

## Generating meshes

Run ``../share/meshes/generate_mesh_dam_break.m`` and change dx = 100, 10, 5, 2, 1.

## Run Simulations

```
mpiexec -n 1 ./ex2b -save true -mesh ../share/meshes/DamBreak_grid5x10.exo -initial_condition ../share/initial_conditions/DamBreak_grid5x10_wetdownstream.IC -dt 0.1 -ts_max_time 72
mpiexec -n 5 ./ex2b -save true -mesh ../share/meshes/DamBreak_grid50x100.exo -initial_condition ../share/initial_conditions/DamBreak_grid50x100_wetdownstream.IC -dt 0.1 -ts_max_time 72
mpiexec -n 5 ./ex2b -save true -mesh ../share/meshes/DamBreak_grid100x200.exo -initial_condition ../share/initial_conditions/DamBreak_grid100x200_wetdownstream.IC -dt 0.1 -ts_max_time 72
mpiexec -n 10 ./ex2b -save true -mesh ../share/meshes/DamBreak_grid250x500.exo -initial_condition ../share/initial_conditions/DamBreak_grid250x500_wetdownstream.IC -dt 0.1 -ts_max_time 72
mpiexec -n 40 ./ex2b -save true -mesh ../share/meshes/DamBreak_grid500x1000.exo -initial_condition ../share/initial_conditions/DamBreak_grid500x1000_wetdownstream.IC -dt 0.1 -ts_max_time 72
```