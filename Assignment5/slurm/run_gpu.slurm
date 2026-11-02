#!/bin/bash
#SBATCH --job-name=chol1013
#SBATCH --partition=khawarizmi--gpu-p
#SBATCH --account=khawarizmi--default-g
#SBATCH --qos=khawarizmi-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=chol1013-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PATH="$HOME/julia-1.12.7/bin:$PATH"
export JULIA_DEPOT_PATH="$HOME/.julia-aarch64:"
export CHOL_MODE="${CHOL_MODE:-explore}"

echo "job ${SLURM_JOB_ID:-local} | node $(hostname) | $(uname -m) | mode $CHOL_MODE | $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv || true

start=$(date +%s)
julia --project=. --startup-file=no cholesky_gpu.jl
echo "julia finished in $(( $(date +%s) - start )) s"
