#!/bin/bash
#SBATCH --mem=64G
#SBATCH --gres=gpu:A6000:4
#SBATCH --output=.slurm_logs/fine_tune_mora_small_fixed.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


# Define an array of model names
names=("410m" "70m")

# Loop through each name
for name in "${names[@]}"
do
    echo "Running with --name $name"
    accelerate launch mora_fine_tune.py --name $name --epochs 8 --lr 1e-4 --rank 8 --dataset_name "alpaca_instruction_gpt4"
done
