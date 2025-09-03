#! /bin/bash

# Modules for compiling on cosma
# Using an old version of HDF5 because of the DMantissa9 lossy filter
module purge
module load gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3

mpic++ -O3 -Wall -Werror -fopenmp -o reduce_snapshots \
  reduce_snapshots_Gas_BH.cpp \
  -lhdf5 -lstdc++fs 2>&1 | tee compile_reduce_snapshots_Gas_BH.log
