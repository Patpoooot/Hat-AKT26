#!/bin/bash
#SBATCH --job-name=chol1013
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=chol1013-%j.out
##SBATCH --partition=FILL_ME
##SBATCH --account=FILL_ME

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PATH="$HOME/.juliaup/bin:$PATH"
export CHOL_MODE="${CHOL_MODE:-explore}"

echo "job ${SLURM_JOB_ID:-local} | node $(hostname) | mode $CHOL_MODE | $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm --format=csv

start=$(date +%s)
julia --project=. --startup-file=no cholesky_gpu.jl
echo "julia finished in $(( $(date +%s) - start )) s"
