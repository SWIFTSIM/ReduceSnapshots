#! /bin/bash

module purge
module load intel_comp/2024.2.0 compiler-rt tbb compiler
module load openmpi/5.0.3
module load ucx/1.13.0rc2
module load parallel_hdf5/1.14.4

mpic++ -O3 -Wall -Werror -qopenmp -o reduce_snapshots \
  reduce_snapshots.cpp \
  -lhdf5 -lstdc++fs 2>&1 | tee compile_reduce_snapshots.log
