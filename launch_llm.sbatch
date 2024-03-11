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
/n/home06/jschwarz/.conda/envs/txgnnv2_env/bin/python3 finetune.py python3 finetune.py --model='dkl_llama2_7b' \
                                                                                       --use_fromage=True \
                                                                                       --n_layers=1 \
                                                                                       --finetune_type='lora' \
                                                                                       --dataset='txgnn_did' \
                                                                                       --n_epochs=30 \
                                                                                       --batch_size=24 \
                                                                                       --learning_rate=0.001 \
                                                                                       --scheduler_type='cosine_decay_with_warmup' \
                                                                                       --valid_every=250 \
                                                                                       --weight_decay=0.01 \
                                                                                       --wandb_track=True
