# AKT_HaT

**Al-Khawarizmi Tour · Online Hackathon HPC & AI · Team Dos-Kallas (#1013)**

Our hackathon work explores air-quality data cleaning, machine learning for PM2.5 prediction, and high-performance numerical computing. Each assignment contains its notebook, supporting material in `Notes/`, and relevant data or outputs.

## Assignments

### 1 · Air-quality data preprocessing — Python

[Open notebook](Assignment1/L1_Air_Quality_Data_Preprocessing_Notebook_Dos-Kallas.ipynb)

Audit and clean Beijing monitoring data: missing values, duplicates, inconsistent labels, units, timestamps and sensor errors. Compare imputation strategies and measure how cleaning affects a virtual PM2.5 sensor using time-based validation.

**Output:** quality analysis, evaluation plots and `beijing_clean_team1013.csv.gz`, used by Assignment 2.

### 2 · AI methods for air quality — Python

[Open notebook](Assignment2/L2_Air_Quality_AI_Methods_Notebook_Dos-Kallas.ipynb)

Compare and tune regression models, classify unhealthy hours (PM2.5 > 75 µg/m³), identify pollution regimes with clustering, build a neural network from scratch, and forecast PM2.5 24 hours ahead.

**Output:** model comparisons, final evaluation and a model card inside the notebook.

### 3 · Cholesky performance — Julia

[Open notebook](Assignment3/L2_Cholesky_Performance_Julia_Notebook.ipynb)

Optimize a Cholesky solver through threading, tiling, mixed precision, memory-access improvements and BLAS/LAPACK. Benchmark speed and numerical accuracy, then explain performance using a roofline model.

**Output:** `results_team1013.json` and `bench_team1013.csv`.
**Status:** baseline results are recorded; optimization tasks and the full benchmark are still in progress.

## Setup

You need **Git**, **Python** (the saved Python notebooks use 3.12), and **JupyterLab**. Assignment 3 also needs **Julia ≥ 1.9** with the **IJulia** kernel. The core assignments run on CPU; a GPU is only relevant to optional bonus work.

```sh
git clone https://github.com/Patpoooot/AKT_HaT.git
cd AKT_HaT
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Install the Python dependencies and start JupyterLab:

```sh
python -m pip install jupyterlab ipykernel numpy pandas matplotlib scipy "scikit-learn>=1.2" joblib
python -m ipykernel install --user --name akt-hat --display-name "Python (AKT_HaT)"
python -m jupyterlab
```

For Assignment 3, run this once in the **Julia REPL**, then refresh JupyterLab:

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
installkernel("Julia (auto threads)", env = Dict("JULIA_NUM_THREADS" => "auto"))
```

Optional Julia plots: `Pkg.add("Plots")`. Without it, the notebook prints tables.

## Running the notebooks

1. Open a notebook from its assignment folder. Select **Python (AKT_HaT)** for Assignments 1–2 or **Julia (auto threads)** for Assignment 3.
2. Keep the notebook's working directory in that assignment folder: data paths and exports are relative. Run cells from top to bottom; keep `TEAM_ID = 1013` and the supplied seeds for reproducibility.
3. For a fresh data pipeline, run Assignment 1 through its export cell, then copy `Assignment1/beijing_clean_team1013.csv.gz` into `Assignment2/`. A cleaned copy is already included there, so Assignment 2 can also run directly.
4. For lighter runs, set `STEP = 3` in Assignment 2 or `QUICK = true` in Assignment 3. Use `STEP = 1` / `QUICK = false` for final runs. Restart the kernel and run all cells before saving final outputs.

## Data & evaluation

- **Source:** UCI Beijing Multi-Site Air Quality; 12 stations, March 2013–February 2017. Compressed raw data is included in Assignments 1–2; no manual extraction is needed.
- **Split:** train on March 2013–February 2016; reserve March 2016–February 2017 for final evaluation. The virtual sensor excludes PM10.
- **Fallbacks:** if local raw data is missing, the loaders try UCI, then synthetic data if downloading fails. Assignment 2 falls back to raw training data if its cleaned input is absent. Check the loader messages before interpreting results.
