# Reproducing the experiment

This guide recreates every number, table and figure in [`results_rtx3050_laptop/`](results_rtx3050_laptop) from scratch: setting up the GPU toolchain, building, testing, getting the datasets, running the benchmark and producing the figures.

Our run used the setup below. Any NVIDIA GPU with CUDA 12 support works; absolute times will differ on other GPUs, so compare the **ordering** of the versions rather than the milliseconds.

| | Our setup |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop GPU (Ampere, sm_86) |
| CPU / RAM | Intel Core i7-13650HX / 16 GB (12 GB given to WSL) |
| OS | Windows 11 + WSL2, Ubuntu 24.04 |
| CUDA | 12.6 |
| Python | 3.12 with numpy, pandas, matplotlib |

Time needed: about 30 minutes of setup, 15–30 minutes of downloads and conversion, and about 15 minutes of benchmarking on our GPU.

---

## Contents
1. [Requirements](#1-requirements)
2. [Set up WSL2 and CUDA (Windows)](#2-set-up-wsl2-and-cuda-windows)
3. [Clone, build and test](#3-clone-build-and-test)
4. [Get the datasets](#4-get-the-datasets)
5. [Run the benchmark](#5-run-the-benchmark)
6. [Produce the figures and tables](#6-produce-the-figures-and-tables)
7. [Compare with our results](#7-compare-with-our-results)
8. [Alternative: Google Colab](#8-alternative-google-colab)
9. [Alternative: native Windows build](#9-alternative-native-windows-build)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

- An **NVIDIA GPU** (compute capability 6.0 or newer) with a current driver. 4 GB of GPU memory is enough for all seven graphs; the largest (`com-Orkut`) needs about 1.3 GB.
- **Linux**, or **Windows with WSL2** (recommended; section 2), or **Google Colab** (section 8).
- **CUDA toolkit 12.x** (`nvcc`), `make`, `g++`.
- **Python 3** with `numpy`, `pandas`, `matplotlib`.
- **RAM:** at least 8 GB available to Linux/WSL for converting the two largest graphs (`com-Orkut`, `soc-LiveJournal1`).
- **Disk:** about 8 GB free for downloads and converted graphs.

---

## 2. Set up WSL2 and CUDA (Windows)

Skip this section on native Linux (install the CUDA toolkit for your distribution instead) or on Colab.

**2.1 Install WSL2 with Ubuntu** (PowerShell as administrator, then restart):
```powershell
wsl --install -d Ubuntu-24.04
```

**2.2 Give WSL enough memory.** By default WSL gets half of your RAM, which is not enough to convert `com-Orkut` on a 16 GB machine. In PowerShell:
```powershell
Set-Content -Path "$env:USERPROFILE\.wslconfig" -Value "[wsl2]`nmemory=12GB"
wsl --shutdown
```
Reopen Ubuntu and check with `free -g`: the `Mem` total should be about 11–12.

**2.3 Install the CUDA toolkit inside Ubuntu.** The Windows NVIDIA driver already exposes the GPU to WSL. **Do not install a Linux NVIDIA driver inside WSL.** Install only the WSL toolkit:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6 make g++ git unzip python3-numpy python3-pandas python3-matplotlib
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

**2.4 Check:**
```bash
nvidia-smi        # must list your GPU
nvcc --version    # must print the CUDA 12.x compiler
```

---

## 3. Clone, build and test

Work inside your Linux home folder (e.g. `~`), not under `/mnt/c/...`: building and reading large files across the Windows boundary is much slower.

```bash
cd ~
git clone https://github.com/md25052007/gpu-cc-ablation.git
cd gpu-cc-ablation
make && make counters
```

`make` builds for the GPU in the machine (`-arch=native`). If your `nvcc` does not support that, pass the architecture explicitly:

| GPU | `ARCH` |
|---|---|
| GTX 10xx (Pascal) | `sm_61` |
| T4, RTX 20xx (Turing) | `sm_75` |
| A100 | `sm_80` |
| RTX 30xx, including the RTX 3050 Laptop (Ampere) | `sm_86` |
| RTX 40xx (Ada) | `sm_89` |

```bash
make ARCH=sm_86 && make counters ARCH=sm_86
```

This produces two programs:
- `bin/cc`: the **timing build**, used for every reported time;
- `bin/cc_counters`: the **counter build**, which also records CAS retries, lane balance and idle threads. It is slower and is never used for timing.

**Run the correctness tests. Do not continue unless they pass.**
```bash
make tests
```
This generates 12 small synthetic test graphs (not benchmark data; see the [README](README.md#correctness-testing)), checks that they match [`tests/expected_test_stats.csv`](tests/expected_test_stats.csv), and runs all five versions on each, comparing every result with a CPU reference. It must end with:
```
ALL TESTS PASSED
```
Our output is in [`tests/gpu_test_log_rtx3050_laptop.txt`](tests/gpu_test_log_rtx3050_laptop.txt). Timings in this test run include GPU start-up costs and are not meaningful; only the `correct` column matters.

Without a GPU you can still check the logic: `make emutests` runs the same kernel code sequentially on the CPU.

---

## 4. Get the datasets

Seven public graphs. **You can either let the script download them (option A) or download them yourself (option B).** Both end with the same `data/<name>.egr` files.

### Option A: automatic

```bash
python3 tools/download.py --list                     # the graphs and their groups
python3 tools/download.py web-Google rgg_n_2_21_s0   # two small ones first, to check the pipeline
python3 tools/download.py                            # all seven
```

For each graph the script:
1. uses a copy you already have in `data/raw/` or in your Downloads folder (Linux `~/Downloads`, or the Windows Downloads folder under WSL), if there is one;
2. otherwise downloads it: SNAP graphs from `snap.stanford.edu`; SuiteSparse graphs from the University of Florida mirror of the collection, then `sparse.tamu.edu`;
3. converts it to `data/<name>.egr`.

If every source fails for a graph, the script names the page to download it from by hand; then use option B for that graph.

### Option B: manual download

Download these files in a browser:

| Graph | Collection | Download (Matrix Market `.tar.gz` / edge list `.txt.gz`) | Page |
|---|---|---|---|
| road_usa | SuiteSparse / DIMACS10 | https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/road_usa.tar.gz | [UF mirror](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/road_usa.html) · [SuiteSparse](https://sparse.tamu.edu/DIMACS10/road_usa) |
| delaunay_n22 | SuiteSparse / DIMACS10 | https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/delaunay_n22.tar.gz | [UF mirror](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/delaunay_n22.html) · [SuiteSparse](https://sparse.tamu.edu/DIMACS10/delaunay_n22) |
| rgg_n_2_21_s0 | SuiteSparse / DIMACS10 | https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/rgg_n_2_21_s0.tar.gz | [UF mirror](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/rgg_n_2_21_s0.html) · [SuiteSparse](https://sparse.tamu.edu/DIMACS10/rgg_n_2_21_s0) |
| kron_g500-logn21 | SuiteSparse / DIMACS10 | https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/kron_g500-logn21.tar.gz | [UF mirror](https://www.cise.ufl.edu/research/sparse/matrices/DIMACS10/kron_g500-logn21.html) · [SuiteSparse](https://sparse.tamu.edu/DIMACS10/kron_g500-logn21) |
| web-Google | SNAP | https://snap.stanford.edu/data/web-Google.txt.gz | [SNAP](https://snap.stanford.edu/data/web-Google.html) |
| soc-LiveJournal1 | SNAP | https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz | [SNAP](https://snap.stanford.edu/data/soc-LiveJournal1.html) |
| com-Orkut | SNAP | https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) |

If a SuiteSparse link does not start a download, open the page and click the **Matrix Market (MM)** link there. (During this project `sparse.tamu.edu` returned "Access Denied" for downloads; the UF mirror served the same files.)

Then either put the files in `data/raw/` (or leave them in your Downloads folder) and run option A, which will find and convert them, or convert them yourself:

```bash
mkdir -p data/raw
cp ~/Downloads/*.tar.gz ~/Downloads/*.txt.gz data/raw/       # under WSL: /mnt/c/Users/<you>/Downloads/

# SuiteSparse graphs (.tar.gz containing <name>/<name>.mtx)
for n in road_usa delaunay_n22 rgg_n_2_21_s0 kron_g500-logn21; do
  tar -xzf data/raw/$n.tar.gz -C data/raw && python3 tools/convert.py data/raw/$n/$n.mtx data/$n.egr
done

# SNAP graphs (.txt.gz edge lists; ids are relabelled to 0..n-1)
python3 tools/convert.py data/raw/web-Google.txt.gz        data/web-Google.egr
python3 tools/convert.py data/raw/soc-LiveJournal1.txt.gz  data/soc-LiveJournal1.egr
python3 tools/convert.py data/raw/com-orkut.ungraph.txt.gz data/com-Orkut.egr
```

### Check the converted graphs

`convert.py` symmetrises the graph and removes duplicate edges and self-loops. It refuses to finish if a Matrix Market file has a different number of entries from its header. Each conversion prints a line like `wrote data/road_usa.egr: |V|=23,947,347 undirected |E|=28,854,312`. The counts must match:

| Graph | \|V\| | undirected \|E\| |
|---|---:|---:|
| road_usa | 23,947,347 | 28,854,312 |
| delaunay_n22 | 4,194,304 | 12,582,869 |
| rgg_n_2_21_s0 | 2,097,152 | 14,487,995 |
| kron_g500-logn21 | 2,097,152 | 91,040,932 |
| web-Google | 875,713 | 4,322,051 |
| soc-LiveJournal1 | 4,847,571 | 42,851,237 |
| com-Orkut | 3,072,441 | 117,185,083 |

`kron_g500-logn21` ends up slightly below its header count because its self-loops are removed; that is expected. Converting `com-Orkut` takes a few minutes and several GB of RAM.

Quick end-to-end check on one graph:
```bash
./bin/cc data/web-Google.egr
```
All five versions must print `yes` in the `correct` column.

---

## 5. Run the benchmark

Before timing: plug a laptop in, set the power mode to best performance, and close games, video and other GPU-heavy programs.

```bash
rm -rf results
python3 tools/bench.py data/*.egr --modes stats,main,counters --runs 5 --tag MyGPU
python3 tools/bench.py data/*.egr --modes sweep-s,sweep-t   --runs 3 --tag MyGPU
```

| Mode | What it runs | Output |
|---|---|---|
| `stats` | graph statistics (dataset table) | `results/stats.csv` |
| `main` | all five versions, automatic `s`, 1 warm-up + 5 timed runs each | `results/timing.csv` |
| `counters` | all five versions with the counter build, 1 run each | `results/counters.csv`, `results/segments.csv` |
| `sweep-s` | V1, V3, V4 for `s` = 1, 2, 4, …, 256 | `results/sweep_s.csv` |
| `sweep-t` | V2, V3, V4 for `t1` ∈ {2, 4, 8} × `t2` ∈ {16, 32, 64, 128} | `results/sweep_t.csv` |

Every run is checked against the CPU reference. If a line says `WRONG PARTITION`, stop: the numbers from that configuration are not valid.

The CSV files are appended to, so delete `results/` before a fresh run. On our GPU both commands together took about 15 minutes; most of that time is loading the graphs and the CPU verification, not the GPU work.

To run a single configuration by hand:
```bash
./bin/cc data/kron_g500-logn21.egr --ver V3,V4 --segments 8 --runs 5
./bin/cc --help
```

---

## 6. Produce the figures and tables

```bash
python3 tools/plots.py                 # reads results/, writes results/figures/
```

| Output | Content |
|---|---|
| `fig1_speedup.png` | speedup over V0, all versions and graphs |
| `fig2_breakdown.png` | setup / Atomic-Hook / Multi-Jump time per version |
| `fig3_sweep_s.png` | time vs number of segments for V1, V3, V4 |
| `fig4_counters.png` | idle threads, SIMT lane efficiency, CAS retry rate |
| `fig5_skew.png` | V4 speedup vs degree skew |
| `fig6_thresholds.png` | V3 / V4 sensitivity to `t1`, `t2` |
| `table_results.csv` / `.tex` | median kernel time per graph and version |
| `table_datasets.csv` / `.tex` | dataset table |
| `summary.csv` | kernel, setup and total medians and all speedups |
| `total/` | Figs. 1, 3, 5 and the results table using total time instead of kernel time |

The headline metric is **kernel time** = Atomic-Hook + Multi-Jump. Setup time (worklists and V4's light-edge list) is shown separately in Fig. 2 and included in the `total/` versions.

---

## 7. Compare with our results

Our run is in [`results_rtx3050_laptop/`](results_rtx3050_laptop). To regenerate its figures from its CSVs:
```bash
python3 tools/plots.py --results results_rtx3050_laptop --out /tmp/our_figures
```

What should carry over to another GPU:
- every run is correct;
- V4 is the fastest variant on the skewed graphs (`kron_g500-logn21`, `web-Google`, `soc-LiveJournal1`, `com-Orkut`), and V3 has far more idle threads than V4 (Fig. 4);
- V3 and V4 are fastest at a segment count well below the automatic one on `com-Orkut` and `kron_g500-logn21` (Fig. 3).

What will differ: absolute times, the exact best `s` and thresholds, and the small margins on the uniform graphs (`road_usa`, `rgg_n_2_21_s0`, `delaunay_n22`). V0 varies by 9–14 % from run to run on the skewed graphs on our GPU, so speedups over V0 move more between runs than comparisons among V1–V4.

---

## 8. Alternative: Google Colab

Open [`notebooks/run_colab.ipynb`](notebooks/run_colab.ipynb) in Colab, select **Runtime → Change runtime type → GPU**, and run the cells top to bottom. The notebook clones this repository, builds for the T4 (`sm_75`), runs the tests, downloads the graphs, runs the full benchmark, shows your figures next to ours and can save the results to Google Drive. A free Colab session should have enough RAM and time for all seven graphs, but may disconnect during long runs; `download.py` skips graphs that are already converted, so you can resume.

---

## 9. Alternative: native Windows build

Requirements: NVIDIA driver, CUDA Toolkit 12.x, Visual Studio 2022 with "Desktop development with C++", and Python with numpy, pandas, matplotlib. Open the **x64 Native Tools Command Prompt for VS 2022**:

```bat
cd gpu-cc-ablation
build.bat                          REM or: build.bat sm_86
python tools\run_tests.py          REM must print ALL TESTS PASSED
python tools\download.py
python tools\bench.py data\road_usa.egr data\delaunay_n22.egr data\rgg_n_2_21_s0.egr data\kron_g500-logn21.egr data\web-Google.egr data\soc-LiveJournal1.egr data\com-Orkut.egr
python tools\plots.py
```
`cmd` does not expand `data\*.egr`, so list the graphs explicitly. Our published run used WSL2, not this path.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` not found in WSL | Update the Windows NVIDIA driver; do not install a Linux driver inside WSL. |
| `nvcc fatal: Unsupported gpu architecture 'compute_native'` | Older `nvcc`: build with `make ARCH=sm_XX` (table in section 3). |
| A download fails with HTTP 403 / "Access Denied" | Use option B: download in a browser from the listed page, put the file in `data/raw/`, rerun. |
| Conversion is killed or very slow | Not enough RAM: raise the WSL memory limit (section 2.2) and close other programs. |
| `warning`/error that the header says N entries but M were read | The download is damaged or truncated: delete it from `data/raw/` and download again. |
| `cannot open data/<name>.egr` | That graph was not converted; rerun `tools/download.py <name>`. |
| `CUDA error: out of memory` | Run one graph at a time and close other GPU programs; `mem_bytes` in `timing.csv` shows what each version needs. |
| Times jump between runs | Laptop on battery or throttling: plug in, use best-performance mode, let it cool, rerun. |
| `WRONG PARTITION` | Do not use those numbers. Rerun `make tests`; if it fails, please open an issue with the output. |
