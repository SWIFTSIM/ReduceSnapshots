#! /bin/bash

# Create reduced snapshots from a SOAP catalogue and a full snapshot

#SBATCH --ntasks 64
#SBATCH --cpus-per-task=2
#SBATCH -o logs/job_reduce.%a.out
#SBATCH -e logs/job_reduce.%a.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 12:00:00
#SBATCH --array=37,77

set -e

module purge
module load gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3

# get the snapshot index from the array task ID
snapnum=`printf '%04d' ${SLURM_ARRAY_TASK_ID}`

# folder where the snapshot files are stored
snapdir="/cosma8/data/dp004/flamingo/Runs/L1000N0900/${SLURM_JOB_NAME}/snapshots"
# folder where the SOAP files are stored
# TODO: Use the flamingo directory
soapdir="/snap8/scratch/dp004/dc-mcgi1/soap_flamingo/Runs/L1000N0900/${SLURM_JOB_NAME}/SOAP_uncompressed/HBTplus"
# output folder for the reduced snapshot
outdir="/snap8/scratch/dp004/dc-mcgi1/reduce_snapshots/L1000N0900/${SLURM_JOB_NAME}"


# full path to the compiled snapshot reduction executable
code=/cosma/home/dp004/${USER}/ReduceSnapshots/reduce_snapshots
# full path to the SOAP halo catalogue
cat="${soapdir}/halo_properties_${snapnum}.hdf5"
# full path to the original snapshot, excluding the .hdf5 and rank extension
# (i.e. flamingo_0077 instead of flamingo_0077.2.hdf5 or flamingo_0077.hdf5)
snap="${snapdir}/flamingo_${snapnum}/flamingo_${snapnum}"
# full path to the membership files, excluding the .hdf5 and rank extension
# TODO: Use soapdir from above
memdir="/cosma8/data/dp004/flamingo/Runs/L1000N0900/${SLURM_JOB_NAME}/SOAP-HBT"
mem="${memdir}/membership_${snapnum}/membership_${snapnum}"
# full path to the output files, excluding the .hdf5 and rank extension
out="${outdir}/flamingo_${snapnum}/flamingo_${snapnum}"

# radius variable inside which particles are kept
# TODO: Change to 100_crit (for testing the N0900 we have halos with npart < 100
# which have no 100_crit values)
RlimVar=SO/200_crit/SORadius

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

# run the software using 16 ranks
# note that the number of ranks cannot exceed the number of snapshot files,
# so 32 is the maximum for an L1000N0900 hydro run
mpirun -np 16 ${code} ${cat} ${snap} ${mem} ${out} \
  ${RlimVar} ${Ncell} ${block_size}
