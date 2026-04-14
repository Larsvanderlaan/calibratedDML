#!/bin/bash

#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=Rout/par-%J.out
#SBATCH --error=Rout/par-%J.err
echo "LOADING R"
module load R
echo "R LOADED"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export CALIBRATEDDML_REPO_ROOT="$REPO_ROOT"
export CDML_PAPER_R_LIBS="${CDML_PAPER_R_LIBS:-$SCRIPT_DIR/.r_libs}"
export CDML_PAPER_RESULTS_DIR="${CDML_PAPER_RESULTS_DIR:-$REPO_ROOT/paper_experiment_results/reproduced}"
cd "$REPO_ROOT"
Rscript -e 'source("paper_experiment_scripts/R_setup.R"); source("paper_experiment_scripts/DRrealdataxgboost.R"); data_name = as.character(Sys.getenv("data_name")); do_real_data(data_name = data_name)'
