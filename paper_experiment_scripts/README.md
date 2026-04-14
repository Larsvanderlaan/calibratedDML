# Legacy Paper Simulations

These scripts reproduce the paper simulations using the original legacy R code in this folder.
They are intentionally isolated from the current R package under `R/` and the Python package under `src/`.

## What This Uses

- `DRkernel.R`: synthetic kernel experiment used for the sample-size plots
- `DRrealdataxgboost.R`: IHDP, ACIC 2018, ACIC 2017, Lalonde, and Twins runs
- `finalized_plots.Rmd`: tables and figures from the generated CSV files

The estimator logic remains in the legacy scripts. The only updates here are:

- repo-local path resolution
- isolated R library handling
- local-result output directories
- cached downloads for external paper datasets
- documented local and Slurm entry points

## Default Locations

- R library: `paper_experiment_scripts/.r_libs`
- generated CSVs and rendered figures: `paper_experiment_results/reproduced`
- cached RealCause downloads: `paper_data/realcause`

Override them with:

- `CDML_PAPER_R_LIBS`
- `CDML_PAPER_RESULTS_DIR`
- `CDML_PAPER_ALLOW_DOWNLOADS`
- `CALIBRATEDDML_REPO_ROOT`

## 1. Install Dependencies

From the repo root:

```bash
Rscript paper_experiment_scripts/install_packages.R
```

This installs the legacy simulation dependencies into the isolated paper library by default.
On macOS, the installer prefers CRAN binaries to avoid unnecessary source builds.
For compatibility with the legacy `sl3` learners, it pins the isolated paper-library `xgboost` to `1.7.11.1`.
It also installs `aciccomp2017` from the `2017/` subdirectory of [`vdorie/aciccomp`](https://github.com/vdorie/aciccomp).

## 2. Exact Reproduction Sequence

From the repo root:

```bash
Rscript paper_experiment_scripts/install_packages.R
bash paper_experiment_scripts/simScriptDRkernel.sh
bash paper_experiment_scripts/simScriptDRrealxgboost.sh
Rscript paper_experiment_scripts/render_finalized_plots.R
```

This writes generated CSVs, rendered HTML, and figures to `paper_experiment_results/reproduced` unless you override `CDML_PAPER_RESULTS_DIR`.

## 3. Quick Smoke Runs

These are the exact smoke commands that were checked in this repo-local legacy setup.

Synthetic kernel experiment:

```bash
CDML_PAPER_NBOOT_KERNEL=50 Rscript -e 'source("paper_experiment_scripts/R_setup.R"); source("paper_experiment_scripts/DRkernel.R"); do_sims(n = 250, nsims = 1)'
```

IHDP real-data benchmark:

```bash
CDML_PAPER_MAX_ITERS=1 CDML_PAPER_NBOOT_REAL=50 Rscript -e 'source("paper_experiment_scripts/R_setup.R"); source("paper_experiment_scripts/DRrealdataxgboost.R"); do_real_data("ihdp")'
```

ACIC 2017 real-data benchmark:

```bash
CDML_PAPER_MAX_ITERS=1 CDML_PAPER_NBOOT_REAL=50 Rscript -e 'source("paper_experiment_scripts/R_setup.R"); source("paper_experiment_scripts/DRrealdataxgboost.R"); do_real_data("acic2017_1")'
```

Expected smoke outputs:

- `paper_experiment_results/reproduced/sim_results_DR_iter=1_n=250_kerneltype=4.csv`
- `paper_experiment_results/reproduced/sim_results_ihdp_xgboost.csv`
- `paper_experiment_results/reproduced/sim_results_acic2017_1_xgboost.csv`

## 4. Full Reproduction Runs

Kernel experiment sweep:

```bash
bash paper_experiment_scripts/simScriptDRkernel.sh
```

Real-data benchmark sweep:

```bash
bash paper_experiment_scripts/simScriptDRrealxgboost.sh
```

Behavior:

- if `sbatch` is available, these submit Slurm jobs
- otherwise they run locally in sequence

Useful overrides:

```bash
NSIMS=100 bash paper_experiment_scripts/simScriptDRkernel.sh
DATA_NAMES="ihdp acic2018_1000 lalonde_cps" bash paper_experiment_scripts/simScriptDRrealxgboost.sh
SAMPLE_SIZES="250 500 1000" bash paper_experiment_scripts/simScriptDRkernel.sh
```

## 5. Render Paper Tables And Figures

After CSV generation:

```bash
Rscript paper_experiment_scripts/render_finalized_plots.R
```

This writes the rendered HTML plus generated PDFs into the configured results directory.

## Notes On External Data

- `ihdp` and `acic2018_*` read from the repo-local `paper_data/`
- `lalonde_cps`, `lalonde_psid`, and `twins` are cached under `paper_data/realcause` and downloaded on first use unless `CDML_PAPER_ALLOW_DOWNLOADS=false`
- `acic2017_*` uses the `aciccomp2017` package from the `2017/` subdirectory of [`vdorie/aciccomp`](https://github.com/vdorie/aciccomp)

## Important Isolation Guarantee

Nothing here routes through the revamped package APIs.
These scripts continue to use the legacy paper estimators defined directly inside `paper_experiment_scripts/`.

## If Installation Still Fails

That points to a machine-level R toolchain issue, not a paper-script issue.
Typical fixes are:

- install Xcode Command Line Tools
- ensure the Homebrew R toolchain is complete
- rerun `Rscript paper_experiment_scripts/install_packages.R`
