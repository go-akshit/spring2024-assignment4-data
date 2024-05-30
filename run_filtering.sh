#!/bin/bash
#SBATCH --job-name=filtering
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=filtering%j.out
#SBATCH --error=filtering%j.err
wget --timeout=5 -i positive_url_sample1.txt --warc-file=subsampled_positive_urls1.warc -O /dev/null