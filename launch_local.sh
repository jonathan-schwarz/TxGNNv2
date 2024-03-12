DATA_SPLIT='complex_disease'

# DKL only
# python3 finetune.py --flagfile=configs/dkl/txgnn_did_default.cfg

# DKL + LLM
python3 finetune.py --flagfile=configs/dkl_llama2_7b/txgnn_did_default.cfg
