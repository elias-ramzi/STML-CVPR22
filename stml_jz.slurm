#!/bin/bash
#SBATCH --job-name=STML     # job name
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name
#SBATCH --array=0

set -x
cd $WORK/STML-CVPR22/

module purge
module load pytorch-gpu/py3/1.11.0

export TORCH_HOME=${SCRATCH}/pretrained_models
export TMPDIR=$JOBSCRATCH
export TQDM_DISABLE='yes'

# size[0]=-1
size[0]=-2


srun python code/main.py \
--gpu-id -1 \
--DATA_DIR $SCRATCH \
--LOG_DIR $SCRATCH/experiments/STML/NO_F_Identity \
--workers 2 \
--model resnet50 \
--embedding_size ${size[${SLURM_ARRAY_TASK_ID}]} \
--optimizer adamp \
--lr 1e-4 \
--dataset SOP \
--view 2 \
--sigma 3 \
--delta 0.9 \
--num_neighbors 2 \
--momentum 0.9 \
--weight_decay 1e-2 \
--emb-lr 1e-2 \
--bn_freeze 1
