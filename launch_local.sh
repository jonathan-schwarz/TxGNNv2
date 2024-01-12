
DATA_SPLIT='complex_disease'

MINE_COMPLEX_DISEASE='/home/jos1479/git/TxGNNv2/checkpoints/mine/complex_disease'

KEXIN_ADRENAL_GLAND='/home/jos1479/git/TxGNNv2/checkpoints/kexin/adrenal_gland'
KEXIN_ANEMIA='/home/jos1479/git/TxGNNv2/checkpoints/kexin/anemia'
KEXIN_CARDIOVASCULAR='/home/jos1479/git/TxGNNv2/checkpoints/kexin/cardiovascular'
KEXIN_CELL_PROLIFERATION='/home/jos1479/git/TxGNNv2/checkpoints/kexin/cell_proliferation'
KEXIN_COMPLEX_DISEASE='/home/jos1479/git/TxGNNv2/checkpoints/kexin/complex_disease'
KEXIN_MENTAL_HEALTH='/home/jos1479/git/TxGNNv2/checkpoints/kexin/mental_health'
KEXIN_RANDOM='/home/jos1479/git/TxGNNv2/checkpoints/kexin/random'

# Pretraining only (debug)
python3 train.py --exp_name='debug_pretrain' --run_pretraining=True --run_finetuning=False --pretrain_batch_size=4 --pretrain_n_epochs=-1 --pretrain_n_steps=25 --n_protos=3 --model_dim=32

# Pretraining only (paper settings) - 2 epochs correspond to 2*6560 steps, just under 3h running time
# python3 train.py --exp_name='paper_pretrain' --run_pretraining=True --run_finetuning=False --pretrain_batch_size=1024 --pretrain_n_epochs=2 --pretrain_n_steps=-1 --n_protos=3 --model_dim=512 --use_wandb=True

# Finetuning only (debug)
# python3 train.py --finetune_n_epochs=2 -exp_name='debug_finetune' --run_pretraining=False --run_finetuning=True --checkpoint=$MINE_COMPLEX_DISEASE --data_split=$DATA_SPLIT
# Finetuning DistMult only (paper settings)
# python3 train.py --finetune_n_epochs=100 -exp_name='paper_distmult_finetune_did' --finetune_dist_mult_only=True --run_pretraining=False --run_finetuning=True  --checkpoint=$MINE_COMPLEX_DISEASE --data_split=$DATA_SPLIT --use_wandb=True
# Finetuning only (paper settings)
# python3 train.py --finetune_n_epochs=500 -exp_name='paper_finetune' --run_pretraining=False --run_finetuning=True  --checkpoint=$MINE_COMPLEX_DISEASE --data_split=$DATA_SPLIT --use_wandb=True

# GP Finetuning only (debug)
# python3 finetune.py --wandb_track=False
