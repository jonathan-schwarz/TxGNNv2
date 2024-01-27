#!/bin/bash
#SBATCH -J finetune_llm                 # job name
#SBATCH -p kempner                      # partition
#SBATCH --account kempner_mzitnik_lab   # partition
#SBATCH -n 1                            # number of nodes
#SBATCH -c 4                            # number of cores
#SBATCH --gres=gpu:1                    # memory in MB
#SBATCH --mem=32G                       # memory in MB
#SBATCH -t 18:00:00                     # time in D-HH:MM

# Load required modules
module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf

# Run jobs
/n/home06/jschwarz/.conda/envs/txgnn_env/bin/python3 finetune_llm.py --model='llama2_7b' --finetune_type='lora' --dataset='txgnn_did' --n_epochs=10 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnn_env/bin/python3 finetune_llm.py --model='llama2_7b' --finetune_type='lora' --dataset='txgnn_dod' --n_epochs=10 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnn_env/bin/python3 finetune_llm.py --model='llama2_7b' --finetune_type='lora' --dataset='txgnn_dcd' --n_epochs=10 --wandb_track=True
