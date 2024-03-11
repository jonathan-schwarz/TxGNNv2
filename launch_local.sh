DATA_SPLIT='complex_disease'

python3 finetune.py --model='dkl' \
                    --dataset='txgnn_did' \
                    --n_epochs=30 \
                    --batch_size=128 \
                    --learning_rate=0.01 \
                    --scheduler_type='valid_plateau' \
                    --valid_every=25 \
                    --wandb_track=True

#python3 finetune.py --model='dkl_llama2_7b' \
#                    --use_fromage=True \
#                    --n_layers=1 \
#                    --finetune_type='lora' \
#                    --dataset='txgnn_did' \
#                    --n_epochs=30 \
#                    --batch_size=24 \
#                    --learning_rate=0.001 \
#                    --scheduler_type='cosine_decay_with_warmup' \
#                    --valid_every=250 \
#                    --weight_decay=0.01 \
#                    --wandb_track=True
