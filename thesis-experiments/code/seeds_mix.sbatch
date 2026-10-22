#!/bin/bash
#SBATCH --job-name=thesis-seeds
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#
# Several configurations spread over both GPUs: workers FIRST..FIRST+2 of TOTAL, three per GPU through MPS.
# The GPU type acts as run-to-run noise (no systematic shift between an RTX 3090 and the H200), and each run
# records its GPU. usage: sbatch --gres=gpu:h200:1 seeds_mix.sbatch 0 6 "c2t c3"
set -euo pipefail
export SEEDS_ROOT=/mnt/beegfs/amughrabi/projects/thesis-seeds
export PY_NEUS2=$SEEDS_ROOT/venv/bin/python PY_MESH=$SEEDS_ROOT/venv/bin/python
export LD_LIBRARY_PATH=/usr/local/spack/0.21.2/opt/spack/linux-rocky9-zen3/gcc-13.2.0/cuda-12.6.0-hpscfrjfydqk46ffaqe7fudhynkie6c4/lib64:${LD_LIBRARY_PATH:-}
FIRST=$1; TOTAL=$2; CONFIGS=$3
export GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
cd $SEEDS_ROOT
echo "host=$(hostname) gpu=$GPU_NAME workers=$FIRST..$((FIRST + 2)) of $TOTAL configs=$CONFIGS"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe-$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control; rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY' EXIT

for w in $FIRST $((FIRST + 1)) $((FIRST + 2)); do
  ( for c in $CONFIGS; do $PY_NEUS2 run_seeds.py --config $c --seeds 1337 1 2 3 4 --shard $w/$TOTAL; done ) > logs/worker_mix_${SLURM_JOB_ID}_$w.log 2>&1 &
done
wait
