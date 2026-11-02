# Week 5 · Green Speed Run · HPC action plan (team 1013)

**Deadline:** Friday 6 November 2026, 04:00 (GMT+3)
**Budget:** 3 GPU-hours in total, at most 5 minutes per job
**Submit:** the performance graph (PNG) + the full path of our Slurm script, e.g. `/home/khawarizmi/team1013/AKT_HaT/Assignment5/run_gpu.slurm`
**Bonus:** least energy used (measured by AutoHPC per job)

## Files in `Assignment5/`

| File | Role |
|---|---|
| `cholesky_gpu.jl` | the GPU tiled Cholesky + benchmark; runs in mode `quick`, `explore` or `final` |
| `run_gpu.slurm` | the job script we submit (and whose path we hand in) |
| `setup_cluster.sh` | one-time setup on the login node: installs Julia 1.13 and the packages, precompiles |
| `Project.toml`, `Manifest.toml` | exact package versions, tested on our laptop |
| `L2_Cholesky_GPU_AutoHPC_Julia_Notebook_Dos-Kallas.ipynb` | reads the cluster CSVs and draws the final graph |

## Phase 0 · Ready before we touch the cluster ✅

- [x] GPU code written and tested on an RTX 3060 (every row passes η ≤ 10·N·ε)
- [x] Three run modes, results saved after every size, stops by itself at 230 s
- [x] GPU energy (J) recorded for every configuration
- [x] Slurm script and setup script written, tested locally
- [ ] Commit and push `Assignment5/` to GitHub (so the cluster can `git clone` it)

## Phase 1 · Get access

1. Find the BrightSkies email with our user name, password and the login address.
2. Log in from a terminal: `ssh <user>@<login-address>`. Change the password if asked.
3. Open the Session 5 slides (Google Drive) and note the **AutoHPC** part: how jobs are submitted, which partition/account to use, how energy is reported.

## Phase 2 · Discover the cluster (login node, costs no GPU time)

Run these and paste the output into the team chat:

```bash
hostname; head -2 /etc/os-release
sinfo -s
sinfo -o "%P %G %l %D"
module avail 2>&1 | head -60
which julia nvidia-smi git curl
curl -sI https://pkg.julialang.org | head -1
df -h $HOME
sacctmgr show assoc user=$USER format=account,partition,qos 2>/dev/null
ls -ld /home/khawarizmi/team1013
```

What we need to learn:

| Question | Where it shows | What we change |
|---|---|---|
| Partition and account names | `sinfo`, `sacctmgr`, slides | fill `#SBATCH --partition` / `--account` in `run_gpu.slurm` |
| GPU model (A100? H100?) | `sinfo` GRES column, then `nvidia-smi` in the quick job | check its peaks in `PEAKS_GFLOPS` against NVIDIA's spec sheet |
| Is Julia provided? | `module avail`, `which julia` | if yes, `module load julia/1.13` in the scripts; if not, `setup_cluster.sh` installs it |
| Internet access | `curl` answers `HTTP/2 200` | if not, we copy the Julia depot over with `scp` |
| How to submit | slides | `sbatch` directly or through AutoHPC |

## Phase 3 · Set up (login node, once)

```bash
cd /home/khawarizmi/team1013
git clone https://github.com/Patpoooot/AKT_HaT.git
cd AKT_HaT/Assignment5
bash setup_cluster.sh
```

Takes about 10–15 minutes (downloads Julia and CUDA.jl, compiles them). No GPU is needed here. If the repository is private, copy the folder instead: `scp -r Assignment5 <user>@<login-address>:/home/khawarizmi/team1013/AKT_HaT/`.

## Phase 4 · Quick test job (~1 minute of GPU)

```bash
sbatch --export=ALL,CHOL_MODE=quick run_gpu.slurm
squeue -u $USER
cat chol1013-<jobid>.out
```

Check in the output:
- the GPU name and driver version (first lines, from `nvidia-smi`)
- `CUDA.versioninfo()` works (CUDA.jl may download its CUDA libraries on this first run)
- every line says `all passed: true`
- `julia finished in … s`: Julia startup time, which every later job pays too

If something fails, fix it and repeat this phase only. Never run `explore` before `quick` passes.

## Phase 5 · Exploration job (≤ 5 minutes of GPU)

```bash
sbatch --export=ALL,CHOL_MODE=explore run_gpu.slurm
```

Tries every size × both precisions × both variants × 7 tile sizes, plus cuSOLVER. Writes `results_gpu_team1013_explore.csv`. Expected 1–3 minutes on a data-center GPU. If the log shows `time budget reached`, run it again for the missing precision only (we split it into two jobs).

Copy the CSV to the laptop and look at it in the notebook:

```bash
scp <user>@<login-address>:/home/khawarizmi/team1013/AKT_HaT/Assignment5/results_gpu_team1013_explore.csv Assignment5/
```

Decide together: best tile size per size and precision, `tile` vs `panel`, and the energy column (`energy_j`).

## Phase 6 · Final lean job (~1 minute of GPU)

```bash
sbatch --export=ALL,CHOL_MODE=final run_gpu.slurm
```

Runs only the fastest configuration per size (read from the exploration CSV) plus cuSOLVER for the dashed reference line. Writes `results_gpu_team1013.csv`. This is the run that produces the graph and the energy number we are judged on.

## Phase 7 · Optional speed-ups (only if Phase 6 is done and budget is left)

1. **TF32 tensor cores** for FP32: one extra exploration, then a new final run.
2. **Mixed precision**: factor in FP32 (or TF32), refine to FP64 accuracy (our HW3 Chapter 3 idea; the Ozaki hint in the brief).

Each needs new code first, tested on the laptop before it goes to the cluster.

## Phase 8 · Graph and submission

1. Copy `results_gpu_team1013.csv` next to the notebook, run it top to bottom.
2. Check the theoretical peaks against NVIDIA's spec sheet for the cluster GPU.
3. Submit `gpu_cholesky_perf_team1013.png` and the path `/home/khawarizmi/team1013/AKT_HaT/Assignment5/run_gpu.slurm`.
4. Commit the CSVs and the graph.

## GPU budget

| Phase | Jobs | GPU minutes (max) |
|---|---|---|
| 4 · quick | 1–2 | 2 |
| 5 · explore | 1–2 | 10 |
| 6 · final | 1 | 5 |
| 7 · speed-ups | 2–4 | 20 |
| reserve | | 143 |
| **total** | | **180** |

## Saving energy

- One `explore` and one `final` job, not many small ones. Each job also pays Julia's startup with the GPU sitting idle.
- Always pass `quick` first, so no job fails halfway.
- Precompile on the login node (Phase 3), never inside a job.
- The final run tests one configuration per size, not the whole grid.
- No interactive GPU sessions left open (`srun --pty`), no `watch nvidia-smi`.
- If startup time turns out large (> 60 s), we can look at precompiling a Julia system image.
