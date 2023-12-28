#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cpp-gen-env

python -m smiles_lstm_hc.goal_directed_generation --model_path smiles_lstm_hc/model_final_0.217.pt --max_len 320 --keep_top 1000 --n_epochs 20 --smiles_file data/prior/prior.csv