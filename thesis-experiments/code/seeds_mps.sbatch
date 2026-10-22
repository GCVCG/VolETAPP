#!/bin/bash
#SBATCH --job-name=thesis-seeds
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#
# Repeatability of NeuS2 volumes across training seeds, N workers sharing one GPU through MPS.
# A single NeuS2 run leaves the H100 and H200 partly idle, and without MPS processes only time-slice.
# usage: sbatch --gres=gpu:h200:1 seeds_mps.sbatch c2 3
set -euo pipefail
export SEEDS_ROOT=/mnt/beegfs/amughrabi/projects/thesis-seeds
export PY_NEUS2=$SEEDS_ROOT/venv/bin/python PY_MESH=$SEEDS_ROOT/venv/bin/python
# CUDA 12.6 runtime NeuS2 was built with (the GPU node has no CUDA libraries on its default path)
export LD_LIBRARY_PATH=/usr/local/spack/0.21.2/opt/spack/linux-rocky9-zen3/gcc-13.2.0/cuda-12.6.0-hpscfrjfydqk46ffaqe7fudhynkie6c4/lib64:${LD_LIBRARY_PATH:-}
CONFIG=$1; N=${2:-3}
cd $SEEDS_ROOT
echo "host=$(hostname) config=$CONFIG workers=$N gpu=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name --format=csv,noheader

# one MPS daemon per job, for the GPU of this job only
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe-$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control; rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY' EXIT

for k in $(seq 0 $((N - 1))); do
  $PY_NEUS2 run_seeds.py --config $CONFIG --seeds 1337 1 2 3 4 --shard $k/$N > logs/worker_${CONFIG}_${SLURM_JOB_ID}_$k.log 2>&1 &
done
wait
