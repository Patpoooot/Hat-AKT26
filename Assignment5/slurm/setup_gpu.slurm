#!/bin/bash
#SBATCH --job-name=setup1013
#SBATCH --partition=khawarizmi--gpu-p
#SBATCH --account=khawarizmi--default-g
#SBATCH --qos=khawarizmi-qos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=00:05:00
#SBATCH --output=setup1013-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PATH="$HOME/julia-1.12.7/bin:$PATH"
export JULIA_DEPOT_PATH="$HOME/.julia-aarch64:"
export JULIA_NUM_PRECOMPILE_TASKS="${JULIA_NUM_PRECOMPILE_TASKS:-8}"
export JULIA_PKG_PRECOMPILE_AUTO=0
DEPOT="$HOME/.julia-aarch64"

echo "job ${SLURM_JOB_ID:-local} | node $(hostname) | $(uname -m) | $(date -Is)"
nvidia-smi --query-gpu=name,driver_version --format=csv || true
julia --version || { echo "julia not found: install the official aarch64 binary in ~/julia-1.12.7"; exit 1; }
find "$DEPOT/compiled" \( -name "*.pidfile" -o -name "jl_*" \) -delete 2>/dev/null

monitor() {
    set +e
    while true; do
        sleep 20
        compiling=$(ps -eo args 2>/dev/null | grep -oE 'compiled/v1\.12/[^/ ]+' | sed 's#compiled/v1.12/##' | sort -u | head -4 | tr '\n' ' ')
        echo "[$(date +%T)] compiled $(du -sh "$DEPOT/compiled" 2>/dev/null | cut -f1) | mem used $(free -g | awk '/Mem:/{print $3}')G | compiling: ${compiling:-none}"
    done
}
monitor &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

start=$(date +%s)
timeout -s INT 270 julia --project=. --startup-file=no -e '
    using Pkg
    t = time(); Pkg.instantiate()
    println("[step] instantiate done in ", round(Int, time() - t), " s")
    t = time(); Pkg.precompile(; timing = true)
    println("[step] precompile done in ", round(Int, time() - t), " s")
    t = time(); using CUDA
    println("[step] using CUDA done in ", round(Int, time() - t), " s")
    CUDA.versioninfo()
    println("CUDA functional: ", CUDA.functional())'
echo "exit code $?"
echo "setup finished in $(( $(date +%s) - start )) s"
echo "next: sbatch --export=ALL,CHOL_MODE=quick run_gpu.slurm"
