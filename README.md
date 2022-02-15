# xpcs-baseline
MD Generated samples to test XPCS algorithms

# Build mdscatter module
 ```shell
 cmake -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/pythonX.Y .
 make
 cp mdscatter.cpython-XY-x86_64-linux-gnu.so ../xpcs
 ```
 
 # Install and run lammps
 Install *lammps* (Ubuntu)
 ```shell
 sudo apt install lammps
 ```
 
Edit input file, if needed, and run lammps

```shell
cd lammps
lmp -in in.melt
```

# Simulate GIXPCS or XPCS
```shell
cd xpcs
python calc_scattering.py
```
or 
```shell
python calc_gisaxs.py
```
Simulation results are saved in a hdf5 file.
