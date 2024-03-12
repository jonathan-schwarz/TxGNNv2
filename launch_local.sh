DATA_SPLIT='complex_disease'

# DKL only
# python3 finetune.py --flagfile=configs/dkl/txgnn_did_default.cfg

# LLM + Linear head
# Baseline (HuggingFace)
# python3 finetune_llm.py
# New
python3 finetune.py --flagfile=configs/mlp_llama2_7b/txgnn_did_linear_no_fromage.cfg

# DKL + LLM
# python3 finetune.py --flagfile=configs/dkl_llama2_7b/txgnn_did_no_fromage.cfg
