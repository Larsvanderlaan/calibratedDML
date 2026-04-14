#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export CALIBRATEDDML_REPO_ROOT="$REPO_ROOT"
export CDML_PAPER_R_LIBS="${CDML_PAPER_R_LIBS:-$SCRIPT_DIR/.r_libs}"
export CDML_PAPER_RESULTS_DIR="${CDML_PAPER_RESULTS_DIR:-$REPO_ROOT/paper_experiment_results/reproduced}"

data_names=(${DATA_NAMES:-acic2018_1000 acic2018_2500 acic2018_5000 acic2018_10000 ihdp lalonde_cps lalonde_psid twins acic2017_17 acic2017_18 acic2017_19 acic2017_20 acic2017_21 acic2017_22 acic2017_23 acic2017_24})

if command -v sbatch >/dev/null 2>&1; then
  for data_name in "${data_names[@]}"; do
    sbatch --export=ALL,data_name="$data_name" "$SCRIPT_DIR/simScriptDRrealxgboost.sbatch"
  done
else
  for data_name in "${data_names[@]}"; do
    (
      cd "$REPO_ROOT"
      data_name="$data_name" Rscript -e 'source("paper_experiment_scripts/R_setup.R"); source("paper_experiment_scripts/DRrealdataxgboost.R"); do_real_data(data_name = Sys.getenv("data_name"))'
    )
  done
fi
