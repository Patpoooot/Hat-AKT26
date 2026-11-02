# Week 5 · Green Speed Run · HPC action plan (team 1013)

**Deadline:** Friday 6 November 2026, 04:00 (GMT+3)
**Budget:** 3 GPU-hours in total, at most 5 minutes per job. Only running jobs count; logging in and working on the login node is free.
**Submit:** the performance graph (PNG) + the full path of our Slurm script: `/home/khawarizmi/team1013/AKT_HaT/Assignment5/slurm/run_gpu.slurm`
**Bonus:** least energy used (measured by AutoHPC per job)

## The cluster (found on 2 November)

| | |
|---|---|
| Login node | `login01.autohpc.ai`, Rocky Linux 9, **x86_64** (browser terminal through Open OnDemand) |
| GPU node | `spark-152`, Ubuntu 24.04, **aarch64 (ARM)**, 18 usable cores, 124 GB, **1× NVIDIA GB10** (DGX Spark, Blackwell) |
| Slurm | partition `khawarizmi--gpu-p`, account `khawarizmi--default-g`, QOS `khawarizmi-qos`, max 5:00 |
| Sharing | one GPU node for **all teams**, each job gets the whole node (`OverSubscribe=EXCLUSIVE`), so jobs may wait in the queue |
| Julia | `/global-apps/spack/software/linux-ubuntu24.04-neoverse_v2/julia-1.12.6/bin` (ARM build, used by our scripts) |
| Internet | GitHub and the Julia package server reachable from the login node |
| Energy | Slurm records watts on `spark-152`; read it after a job with `sacct` (below) |

Consequences:
- Login node and GPU node have **different processors**: Julia packages must be installed and compiled **on the GPU node**, which is what `slurm/setup_gpu.slurm` does. We keep them in their own folder, `~/.julia-aarch64`.
- The GB10 runs FP64 at about 1/64 of its FP32 speed and has no FP64 tensor cores. FP64 will be far below FP32 on the graph. This is where the brief's Ozaki hint matters (FP64 accuracy from fast low-precision tensor cores).
- Never start an interactive GPU app in Open OnDemand (Jupyter, desktop, interactive session): it holds the only GPU and its time keeps counting until it is closed. Only submit batch jobs.

## Files in `Assignment5/`

| File | Role |
|---|---|
| `src/cholesky_gpu.jl` | the GPU tiled Cholesky + benchmark; modes `quick`, `explore`, `final` |
| `src/prefetch_artifacts.jl` | downloads the aarch64 CUDA artifacts ahead of time |
| `slurm/run_gpu.slurm` | the job script we submit (and whose path we hand in) |
| `slurm/setup_gpu.slurm` | one-off job: installs and compiles our Julia packages on the ARM GPU node |
| `results/` | final CSV; `results/exploration/` holds the quick, explore and local runs |
| `figures/` | the performance graph |
| `Project.toml`, `Manifest-v1.12.toml` | exact package versions for the cluster's Julia 1.12 (`Manifest.toml` is the same for Julia 1.13 on the laptop) |
| `L2_Cholesky_GPU_AutoHPC_Julia_Notebook_Dos-Kallas.ipynb` | reads the cluster CSVs and draws the final graph |

## Phase 0 · Ready before the cluster ✅
- [x] GPU code tested on an RTX 3060 with Julia 1.13 and 1.12 (every row passes η ≤ 10·N·ε)
- [x] Three run modes, results saved after every size, stops by itself at 230 s
- [x] GPU energy (J) recorded for every configuration
- [x] Slurm scripts with the real partition, account and Julia path
- [ ] Commit and push, then `git pull` on the cluster

## Phase 1 · Access ✅
Browser terminal on `login01.autohpc.ai` as `team1013_khawarizmi`.

## Phase 2 · Discovery ✅
Done (table above). A teammate already ran `~/hw5/test_gpu.slurm` (jobs 1605, 1606, 1 s each): check with the team before running more test jobs.

## Phase 3 · Get the code and set up (one job, ≤ 5 min)

On the login node:
```bash
cd /home/khawarizmi/team1013
git clone https://github.com/Patpoooot/AKT_HaT.git
cd AKT_HaT/Assignment5
sbatch slurm/setup_gpu.slurm
squeue -u $USER
cat setup1013-<jobid>.out
```
Success = the log ends with `CUDA functional: true` and `setup finished in … s`. If the job hit the 5-minute limit while compiling, submit it once more: it continues where it stopped. If it fails on a download, the GPU node has no internet; then we download on the login node instead (ask before trying anything else).

## Phase 4 · Quick test (~1 min)
```bash
sbatch --export=ALL,CHOL_MODE=quick slurm/run_gpu.slurm
```
Check the log `chol1013-<jobid>.out`:
- GPU name `NVIDIA GB10`, `aarch64`
- every line ends with `all passed: true`
- `julia finished in … s`: the startup cost every later job pays
- the theoretical peaks printed for the GB10: compare with NVIDIA's DGX Spark spec sheet before trusting the graph

## Phase 5 · Exploration (≤ 5 min)
```bash
sbatch --export=ALL,CHOL_MODE=explore slurm/run_gpu.slurm
```
Every size × FP32/FP64 × both variants × 7 tile sizes, plus cuSOLVER → `results/exploration/results_gpu_team1013_explore.csv`. If the log says `time budget reached`, run the missing precision in a second job. Then copy the CSV to the laptop (download it from the Open OnDemand file browser, or commit it from the cluster) and decide together: best tile size, `tile` vs `panel`, energy per configuration.

## Phase 6 · Final lean run (~1 min)
```bash
sbatch --export=ALL,CHOL_MODE=final slurm/run_gpu.slurm
```
Only the fastest configuration per size + cuSOLVER → `results/results_gpu_team1013.csv`. This run produces the graph and the energy number we are judged on.

## Phase 7 · Optional speed-ups (only after Phase 6, budget permitting)
1. TF32 tensor cores for FP32
2. Mixed precision / Ozaki-style FP64 from low-precision tensor cores (the brief's hint; the GB10's weak FP64 makes this the big win)

Each one: new code, tested on the laptop, then one exploration and one final job.

## Phase 8 · Graph and submission
1. Put `results_gpu_team1013.csv` in `results/` and run the notebook top to bottom.
2. Check the GB10 peaks against the spec sheet.
3. Submit `figures/gpu_cholesky_perf_team1013.png` and `/home/khawarizmi/team1013/AKT_HaT/Assignment5/slurm/run_gpu.slurm`.
4. Commit the CSVs and the graph.

## Checking time and energy after every job
```bash
sacct -j <jobid> -o JobID,JobName,Elapsed,State,ConsumedEnergy,AveWatts
sacct -u $USER -S 2026-11-01 -o JobID,JobName,Elapsed,State,ConsumedEnergy
squeue -u $USER
```
`squeue` empty = nothing of ours is running, nothing is being counted.

## GPU budget

| Phase | Jobs | GPU minutes (max) |
|---|---|---|
| 3 · setup | 1–2 | 10 |
| 4 · quick | 1 | 2 |
| 5 · explore | 1–2 | 10 |
| 6 · final | 1 | 5 |
| 7 · speed-ups | 2–4 | 20 |
| reserve | | 133 |
| **total** | | **180** |

## Saving energy
- One `explore` and one `final` job, not many small ones: each job also pays Julia's startup.
- Always pass `quick` before `explore`, so no long job fails halfway.
- Compile once in `slurm/setup_gpu.slurm`, never inside a benchmark job.
- The final run tests one configuration per size, not the whole grid.
- No interactive GPU sessions, no `watch nvidia-smi`.
- Submit early: one GPU for all teams means long queues near the deadline.
