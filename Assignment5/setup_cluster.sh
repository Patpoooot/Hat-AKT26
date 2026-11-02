#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v julia >/dev/null 2>&1 && [ ! -x "$HOME/.juliaup/bin/julia" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.13
fi
export PATH="$HOME/.juliaup/bin:$PATH"

julia --version
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA; println("CUDA.jl loaded, functional GPU here: ", CUDA.functional())'
echo "setup done: submit with  sbatch --export=ALL,CHOL_MODE=quick run_gpu.slurm"
