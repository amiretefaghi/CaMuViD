#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=distributed_training
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=amir.etefaghidar@ufl.edu
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1   
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem-per-cpu=6GB
#SBATCH --constraint=a100

# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)
module purge
module load conda/24.3.0
conda activate internimage

# PRINTS
#=======
date; pwd; which python
export HOST=$(hostname -s)
NODES=$(scontrol show hostnames | grep -v $HOST | tr '\n' ' ')
echo "Host: $HOST" 
echo "Other nodes: $NODES"

# Setting up paths and variables
#===============================
echo "Starting $SLURM_GPUS_PER_TASK process(es) on each node..."

PYTHON=${PYTHON:-"python3"}

DATE=$(date '+%d-%b') 

# PYTHON SCRIPT
#==============
echo "==========Starting Distributed Training============="
srun $PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=63668 --use_env DuViDA.py 
echo "==========Training Complete============="
