#!/bin/bash
#SBATCH --job-name=mpi-test
#SBATCH --account=cs402users
#SBATCH --partition=cs402-viper
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:00:30

module purge
module load cs402-mpi

mpirun ./karman
