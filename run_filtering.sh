#!/bin/bash
#SBATCH --job-name=filtering
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=filtering%j.out
#SBATCH --error=filtering%j.err
python3 cs336-data/cs336_data/cc_filtering.py