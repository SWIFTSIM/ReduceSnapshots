#! /bin/bash

# Create reduced snapshots from a SOAP catalogue and a full snapshot
# We need to run on 2 nodes to make everything fit in memory

#SBATCH --ntasks 64
#SBATCH --cpus-per-task=4
#SBATCH -J L1000N3600_HYDRO_FIDUCIAL-reduced-snapshots
#SBATCH -o logs/job_reduce.%a.out
#SBATCH -e logs/job_reduce.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
#SBATCH --array=37,77

module purge
module load cosma/2018 \
  utils/201805 \
  intel_comp/2020-update2 \
  intel_mpi/2020-update2 \
  ucx/1.8.1 parmetis/4.0.3-64bit \
  parallel_hdf5/1.10.6 gsl/2.5 \
  fftw/3.3.8 cmake

# get the snapshot index from the array task ID
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# folder where the snapshot files are stored
snapdir="/cosma8/data/dp004/flamingo/Runs/L1000N3600/HYDRO_FIDUCIAL/snapshots"
# folder where the SOAP files are stored
soapdir="/snap8/scratch/dp004/dc-mcgi1/reduce_snapshots/L1000N3600/HYDRO_FIDUCIAL/SOAP/VR"
# output folder for the reduced snapshot
outdir="/snap8/scratch/dp004/dc-mcgi1/reduce_snapshots/L1000N3600/HYDRO_FIDUCIAL/pr_snapshots_reduced"

# full path to the compiled snapshot reduction executable
code=/cosma/home/dp004/dc-mcgi1/ReduceSnapshots/reduce_snapshots_noSOlist
# full path to the SOAP halo catalogue
cat="${soapdir}/halo_properties_${snapnum}.hdf5"
# full path to the original snapshot, excluding the .hdf5 and rank extension
# (i.e. flamingo_0077 instead of flamingo_0077.2.hdf5 or flamingo_0077.hdf5)
snap="${snapdir}/flamingo_${snapnum}/flamingo_${snapnum}"
# full path to the membership files, excluding the .hdf5 and rank extension
mem="${soapdir}/membership_${snapnum}/membership_${snapnum}"
# full path to the output files, excluding the .hdf5 and rank extension
out="${outdir}/flamingo_${snapnum}/flamingo_${snapnum}"

# mass limit above which halos are kept, in log10(M/Msun)
logMlim=12.5
# mass variable used in the mass cut
MlimVar=SO/200_crit/TotalMass
# radius variable inside which particles are kept
RlimVar=SO/50_crit/SORadius

# number of SWIFT cells that is processed in one go
# a larger number is better for performance, while a smaller number is better
# for memory usage
Ncell=1024
# HDF5 block size for copying operations
# a larger number is better for performance, but uses more memory
# on cosma 8, this can be set to 18446744073709551615, which is basically
# unlimited
block_size=18446744073709551615

# prevent MPI ranks from busy-looping in barriers
# this reduces the system load during periods of imbalances and makes more
# threads available for OpenMP
export I_MPI_WAIT_MODE=1
export I_MPI_THREAD_YIELD=2

# run the software using 64 ranks
# note that the number of ranks cannot exceed the number of snapshot files,
# so 64 is the maximum for an L1000N3600 hydro run
mpirun -np 64 ${code} ${cat} ${snap} ${mem} ${logMlim} ${out} ${MlimVar} \
  ${RlimVar} ${Ncell} ${block_size}
