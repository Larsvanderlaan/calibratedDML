#!/bin/bash
#SBATCH --job-name LarsJob%jls        # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short       # Slurm partition to use
#SBATCH --ntasks 1            # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-00:20          # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=1000       # Memory limit for each tasks (in MB)
#SBATCH -o ./%j.out      # File to which STDOUT will be written
#SBATCH -e ./%j.err      # File to which STDERR will be written
#SBATCH --mail-type=NONE        # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=lvdlaan@uw.edu # Email to which notifications will be sent
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export CALIBRATEDDML_REPO_ROOT="$REPO_ROOT"
export CDML_PAPER_R_LIBS="${CDML_PAPER_R_LIBS:-$SCRIPT_DIR/.r_libs}"
module load R
Rscript "$SCRIPT_DIR/install_packages.R"
