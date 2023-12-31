
##==== For test purposes ====
## SMILES GA
# python -m smiles_ga.goal_directed_generation --smiles_file data/prior/prior_tmp.csv --n_jobs 110 --generations 5 --n_mutations 10 --population_size 30 --gene_size 30 

## Graph GA
# python -m graph_ga.goal_directed_generation --smiles_file data/prior/prior.csv --n_jobs 110 --generations 5 --population_size 30 --offspring_size 30 --random_start  

## Frag GA
# python -m frag_gt.goal_directed_generation --fragstore_path frag_gt/data/fragment_libraries/guacamol_v1_all_fragstore_brics.pkl --smiles_file data/prior/prior.csv  --generations 5 --population_size 30 --random_start

# Graph MCTS
# python -m graph_mcts.analyze_dataset --smiles_file data/prior/prior.csv
# python -m graph_mcts.goal_directed_generation --generations 5 --n_jobs 110 --population_size 30 --max_atoms 200

## SMILES LSTM Hill Climbing
# python -m smiles_lstm_hc.train_smiles_lstm_model --train_data data/prior/prior.csv --valid_data data/prior/prior.csv --max_len 320 --n_epochs 100
# python -m smiles_lstm_hc.goal_directed_generation --model_path smiles_lstm_hc/model_final_0.217.pt --max_len 320 --keep_top 30 --n_epochs 5 --smiles_file data/prior/prior.csv --random_start 

##==== For full run ====
## SMILES GA
python -m smiles_ga.goal_directed_generation --smiles_file data/prior/prior.csv --n_jobs 110 --generations 500

## Graph GA
# python -m graph_ga.goal_directed_generation --smiles_file data/prior/prior.csv --n_jobs 110 --generations 500

## Frag GA
# python -m frag_gt.goal_directed_generation --fragstore_path frag_gt/data/fragment_libraries/guacamol_v1_all_fragstore_brics.pkl --smiles_file data/prior/prior.csv --generations 500

## Graph MCTS
# python -m graph_mcts.goal_directed_generation --generations 5 --n_jobs 110 --max_atoms 200

## SMILES LSTM Hill Climbing
# python -m smiles_lstm_hc.goal_directed_generation --model_path smiles_lstm_hc/model_final_0.217.pt --max_len 320 --keep_top 1000 --n_epochs 20 --smiles_file data/prior/prior.csv
