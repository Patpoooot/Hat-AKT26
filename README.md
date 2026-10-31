# AKT_HaT

**Al-Khawarizmi Tour · HPC & AI Hackathon**

Team **Dos-Kallas** · #1013

Our work on air-quality data, machine learning, and high-performance computing.

## Assignments

| # | Topic | Tools | Open |
|---|---|---|---|
| 1 | Clean and prepare Beijing air-quality data | Python | [Notebook](Assignment1/L1_Air_Quality_Data_Preprocessing_Notebook_Dos-Kallas.ipynb) |
| 2 | Predict PM2.5 with machine learning | Python | [Notebook](Assignment2/L2_Air_Quality_AI_Methods_Notebook_Dos-Kallas.ipynb) |
| 3 | Optimize and benchmark a Cholesky solver | Julia | [Notebook](Assignment3/L2_Cholesky_Performance_Julia_Notebook_Dos-Kallas.ipynb) |
| 4 | GPU programming: Hello World, AXPY, dot product, GER, and GEMM | CUDA C++ | [Code](Assignment4/akt26_gpu_dev/) · [Slides](Assignment4/Notes/akt26_gpu_prog.pdf) |

Each assignment has supporting material in its `Notes/` folder.

## Getting started

```sh
git clone https://github.com/Patpoooot/AKT_HaT.git
cd AKT_HaT
```

### Python · Assignments 1–2

Use Python 3.12 and create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

```sh
python -m pip install jupyterlab ipykernel numpy pandas matplotlib scipy "scikit-learn>=1.2" joblib
python -m jupyterlab
```

Open a notebook, select the Python kernel, and run cells from top to bottom with the working directory set to its assignment folder.

Assignment 2 includes cleaned data and can run directly. To use a fresh export from Assignment 1, copy `beijing_clean_team1013.csv.gz` into `Assignment2/`.

### Julia · Assignment 3

Install Julia 1.9 or newer and JupyterLab. Run this once in the Julia REPL:

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
installkernel("Julia (auto threads)", env = Dict("JULIA_NUM_THREADS" => "auto"))
```

Open the notebook in JupyterLab and select **Julia (auto threads)**. Use `QUICK = true` for a shorter benchmark. Install `Plots` with `Pkg.add("Plots")` if you want plots.

### CUDA · Assignment 4

You need an NVIDIA GPU, the CUDA Toolkit (`nvcc` and cuBLAS), a compatible C++ compiler, and Make.

Start with Hello World from the repository root:

```sh
cd Assignment4/akt26_gpu_dev/0-hello-world
make
./hello-world
```

On Windows, run `.\hello-world.exe`. Continue through `1-axpy`, `2-dot`, `3-ger`, and `4-gemm`; each folder has its own makefile. The default GPU target is `sm=80`; use `make sm=<target>` to select your GPU's compute capability.

## Data & license

Assignments 1–2 use the **UCI Beijing Multi-Site Air Quality** dataset. Compressed data is included; check notebook loader messages, as missing data can trigger a synthetic fallback.

This project is available under the [MIT License](LICENSE).
