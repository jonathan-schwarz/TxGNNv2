DATA_SPLIT='complex_disease'

python3 finetune.py --model='dkl' \
     --dataset='txgnn_did' \
     --n_epochs=30 \
     --batch_size=128 \
     --learning_rate=2e-5 \
     --scheduler_type='cosine_decay_with_warmup' \
     --weight_decay=0.01 \
     --wandb_track=True
