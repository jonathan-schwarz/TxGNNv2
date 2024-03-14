#!/bin/bash
#SBATCH -J finetune_llm_txgnn           # job name
#SBATCH -p kempner                      # partition
#SBATCH --account kempner_mzitnik_lab   # partition
#SBATCH -n 1                            # number of nodes
#SBATCH -c 4                            # number of cores
#SBATCH --gres=gpu:1                    # memory in MB
#SBATCH --mem=256G                      # memory in MB
#SBATCH -t 72:00:00                     # time in D-HH:MM

# Load required modules
module load ncf/1.0.0-fasrc01
module load miniconda3/py39_4.11.0-linux_x64-ncf

# Run jobs

# LLM + Linear head
# Baseline (HuggingFace)
# /n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune_llm.py
# New
/n/home06/jkondic/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/mlp_llama2_7b/txgnn_did_linear_no_fromage.cfg

# DKL + LLM
# /n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --flagfile=configs/dkl_llama2_7b/txgnn_did_default.cfg
