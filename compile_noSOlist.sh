#! /bin/bash

if true ; then
  module purge
  module load intel_comp/2020-update2 \
    intel_mpi/2020-update2 \
    ucx/1.8.1 parmetis/4.0.3-64bit \
    parallel_hdf5/1.10.6 gsl/2.5 \
    fftw/3.3.8 cmake

  mpiicpc -O3 -qopenmp -qoverride-limits \
    -I/cosma/local/parallel-hdf5//intel_2020-update2_intel_mpi_2020-update2/1.10.6/include \
    -o reduce_snapshots_noSOlist reduce_snapshots_noSOlist.cpp \
    -L/cosma/local/parallel-hdf5//intel_2020-update2_intel_mpi_2020-update2/1.10.6/lib \
    -Wl,-rpath=/cosma/local/parallel-hdf5//intel_2020-update2_intel_mpi_2020-update2/1.10.6/lib \
    -lhdf5 -lstdc++fs 2>&1 | tee compile.log
else
  module load OpenMPI HDF5

  mpic++ -O3 -Wall -Werror -fopenmp -o reduce_snapshots_noSOlist \
    reduce_snapshots_noSOlist.cpp \
    -lhdf5 -lstdc++fs 2>&1 | tee compile_noSOlist.log
fi
