#!/bin/bash
#SBATCH --job-name=bayesianCART_tau
#SBATCH --mail-user=yilingh@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --array=1-20%20
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01-00:00:00
#SBATCH --output=bcart_tau-%A_%a.out

set -e -o pipefail
cd /home/yilingh/SI-CART/Simulations
module load python/3.13

# Each array task launches one serial Python process on one CPU.
# Keep using the Python environment that ran the preceding BCART experiment.
range_per_job=5
start=$(( (SLURM_ARRAY_TASK_ID - 1) * range_per_job ))
end=$(( SLURM_ARRAY_TASK_ID * range_per_job ))

# Explicit ranges override Python's automatic 50-replication array default.
# Separate prefix keeps this tau sweep distinct from the earlier beta experiment.
python3 -u bayesianCART.py bcart_tau --start "$start" --end "$end" \
    --betas '[2]' --taus '[1,2,5,10]'
