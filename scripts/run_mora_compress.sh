#!/bin/bash
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=.slurm_logs/trunc_eval_mora_small.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


accelerate launch mora_compress.py