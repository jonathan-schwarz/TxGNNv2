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

# DistMult
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/distmult/txgnn_did_default.cfg
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/distmult/txgnn_dod_default.cfg
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/distmult/txgnn_dcd_default.cfg

# MLP
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/mlp/txgnn_did_default.cfg
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/mlp/txgnn_dod_default.cfg
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/mlp/txgnn_dcd_default.cfg

# DKL
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/dkl/txgnn_did_default.cfg
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/dkl/txgnn_dod_default.cfg
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/dkl/txgnn_dcd_default.cfg
