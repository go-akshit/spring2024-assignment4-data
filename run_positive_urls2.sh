#!/bin/bash
#SBATCH --job-name=positive_urls2
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=positive_urls2%j.out
#SBATCH --error=positive_urls2%j.err
wget --timeout=5 -i positive_url_sample2.txt --warc-file=subsampled_positive_urls2.warc -O /dev/null