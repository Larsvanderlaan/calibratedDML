#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export CALIBRATEDDML_REPO_ROOT="$REPO_ROOT"
export CDML_PAPER_R_LIBS="${CDML_PAPER_R_LIBS:-$SCRIPT_DIR/.r_libs}"
export CDML_PAPER_RESULTS_DIR="${CDML_PAPER_RESULTS_DIR:-$REPO_ROOT/paper_experiment_results/reproduced}"
export NSIMS="${NSIMS:-5000}"

sample_sizes=(${SAMPLE_SIZES:-100 250 500 750 1000 2000 3000 4000 5000 7500 9000 12000})

if command -v sbatch >/dev/null 2>&1; then
  for n in "${sample_sizes[@]}"; do
    sbatch --export=ALL,n="$n" "$SCRIPT_DIR/simScriptDRkernel.sbatch"
  done
else
  for n in "${sample_sizes[@]}"; do
    (
      cd "$REPO_ROOT"
      n="$n" Rscript -e 'source("paper_experiment_scripts/R_setup.R"); source("paper_experiment_scripts/DRkernel.R"); do_sims(n = as.numeric(Sys.getenv("n")), nsims = as.numeric(Sys.getenv("NSIMS")))' 
    )
  done
fi
