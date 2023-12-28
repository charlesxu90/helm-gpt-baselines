#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/work/helm-gpt-baselines/env

## SMILES GA
python -m smiles_ga.goal_directed_generation --smiles_file data/prior/prior.csv --n_jobs 110 --generations 5 --n_mutations 10 --population_size 30 --gene_size 30 --random_start  

# python -m smiles_ga.goal_directed_generation --smiles_file data/prior/prior.csv --n_jobs 90 --generations 500

## Graph GA
# python -m graph_ga.goal_directed_generation --smiles_file data/prior/prior.csv --n_jobs 90 --generations 500

## Frag GT
# python -m frag_gt.goal_directed_generation --fragstore_path frag_gt/data/fragment_libraries/guacamol_v1_all_fragstore_brics.pkl --smiles_file data/prior/prior.csv --generations 500

## Graph MCTS
# python -m graph_mcts.goal_directed_generation --generations 5 --n_jobs 90 --max_atoms 500