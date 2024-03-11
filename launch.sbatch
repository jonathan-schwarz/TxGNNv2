#!/bin/bash
#SBATCH -J finetune_txgnn               # job name
#SBATCH -p kempner                      # partition
#SBATCH --account kempner_mzitnik_lab   # partition
#SBATCH -n 1                            # number of nodes
#SBATCH -c 4                            # number of cores
#SBATCH --gres=gpu:1                    # memory in MB
#SBATCH --mem=256G                      # memory in MB
#SBATCH -t 12:00:00                     # time in D-HH:MM

# Load required modules
module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf

# Run jobs
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='distmult' --dataset='txgnn_did' --n_epochs=30 --batch_size=128 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='distmult' --dataset='txgnn_dod' --n_epochs=30 --batch_size=128 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='distmult' --dataset='txgnn_dcd' --n_epochs=30 --batch_size=128 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True

/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='mlp' --dataset='txgnn_did' --n_epochs=30 --batch_size=128 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='mlp' --dataset='txgnn_dod' --n_epochs=30 --batch_size=128 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='mlp' --dataset='txgnn_dcd' --n_epochs=30 --batch_size=128 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True

/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='dkl' --dataset='txgnn_did' --n_epochs=30 --batch_size=128 --weight_decay=1e-4 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='dkl' --dataset='txgnn_dod' --n_epochs=30 --batch_size=128 --weight_decay=1e-4 --learning_rate=0.01 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py --model='dkl' --dataset='txgnn_dcd' --n_epochs=30 --batch_size=128 --weight_decay=1e-4 --learning_rate=0.03 --scheduler='valid_plateau' --valid_every=25 --wandb_track=True
