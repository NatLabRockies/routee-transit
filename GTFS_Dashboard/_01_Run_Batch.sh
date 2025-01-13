#!/bin/bash
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=04:00:00   # Required, test with debug node for 1 hour, switch to 4 hours or logner for full run
#SBATCH --account=mbap ##
#SBATCH --mail-user=zliu2@nrel.gov ## Change to your account
#SBATCH --job-name=FTA_TA
#SBATCH --output=GTFS_Whole.out
#SBATCH --error=GTFS_Whole.err
#SBATCH --partition=short
#SBATCH --qos=high

module load anaconda3
conda activate bosch_phaseii

python _01_Trip_Feature_for_Energy_Estimation.py

