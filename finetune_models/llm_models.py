'''LLM Models.'''
import evaluate
import numpy as np
import torch

from pynvml import *
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from .custom_llama import LlamaForCustomSequenceClassification
from .models import FeatureExtractor
from peft import LoraConfig

accuracy = evaluate.load('accuracy')

ID2LABEL = {0: 'NEGATIVE', 1: 'POSITIVE'}
LABEL2ID = {'NEGATIVE': 0, 'POSITIVE': 1}


CACHE_PATH = '/n/holyscratch01/mzitnik_lab/jschwarz/cache/'
# Only use this in case of any network problems
#CACHE_PATH = '/n/home06/jschwarz/cache'


def _get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f'GPU memory occupied: {info.used//1024**2} MB.'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    full_pred_probs = softmax(logits, axis=1)
    pred_probs = full_pred_probs[:, 1]
    pred_label = np.argmax(logits, axis=1)

    correct_prediction_probs = 1.0 - full_pred_probs[
        range(labels.shape[0]), labels.astype(np.int32).flatten()][:, np.newaxis]

    metric_dict = accuracy.compute(predictions=pred_label, references=labels)
    metric_dict['auroc'] = roc_auc_score(labels, pred_probs)
    metric_dict['auprc'] = average_precision_score(labels, pred_probs)
    metric_dict['auroc_auprc'] = metric_dict['auroc'] * metric_dict['auprc']
    return metric_dict


def get_tokenizer(model_type):
    # Load relevant tokenizer
    if 'distilbert' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased', cache_dir=CACHE_PATH)
    elif 'mixtral' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(
            'mistralai/Mixtral-8x7B-v0.1', cache_dir=CACHE_PATH)
        # TODO(schwarzjn): Double check
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # Fix weird overflow issue with fp16 training
    elif  'llama2_7b' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(
            'NousResearch/Llama-2-7b-hf', trust_remote_code=True, cache_dir=CACHE_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # Fix weird overflow issue with fp16 training
    elif  'llama2_13b'in model_type:
        tokenizer = AutoTokenizer.from_pretrained(
            'NousResearch/Llama-2-13b-hf', trust_remote_code=True, cache_dir=CACHE_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # Fix weird overflow issue with fp16 training
    elif  'llama3_8b' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Meta-Llama-3-8B', trust_remote_code=True, cache_dir=CACHE_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right' # Fix weird overflow issue with fp16 training

    return tokenizer


def get_llm(model_type, task_type, tokenizer, use_4bit=True):
    del task_type  # Currently unused

    # Load the entire model on the GPU 0
    # TODO(schwarzjn): Fix for multi-GPU jobs
    device_map = {'': 0}

    # Check GPU compatibility with bfloat16
    use_bf16 = False

    compute_dtype = getattr(torch, 'float16')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Check if we can use bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print('=' * 80)
            print('Your GPU supports bfloat16: accelerating training with use_bf16=True')
            print('=' * 80)
            use_bf16 = True
    if 'distilbert' == model_type:
        # No quantization necessary for DistilBert (small model)
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            device_map=device_map,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=CACHE_PATH,
        )
    elif 'mixtral' in model_type:
        model = AutoModelForSequenceClassification.from_pretrained(
            'mistralai/Mixtral-8x7B-v0.1',
            quantization_config=bnb_config,
            device_map=device_map,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=CACHE_PATH,
        )
    elif 'llama' in model_type:

        if'llama2_7b' in model_type:
            model_name = 'NousResearch/Llama-2-7b-hf'
        elif'llama2_13b' in model_type:
            model_name = 'NousResearch/Llama-2-13b-hf'
        elif'llama3_8b' in model_type:
            model_name = 'meta-llama/Meta-Llama-3-8B'

        # Custom version
        model = LlamaForCustomSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=CACHE_PATH,
        )
        model.config.pad_token_id = model.config.eos_token_id

        # Activate the more accurate but slower computation of the linear layers
        model.config.pretraining_tp = 1

    model.config.use_cache = False

    print('=' * 80)
    print('Model loaded. ' + _get_gpu_utilization())
    print('=' * 80)

    return model, use_bf16


def get_peft(model, task_type, finetune_type, lora_apply_everywhere, use_final_layer=True):
    optim = 'adamw_torch'
    if 'lora' in finetune_type:

        kwargs = {
            'task_type': task_type,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'r': 64,
            'bias': 'none',
        }

        if not lora_apply_everywhere:
            kwargs['target_modules'] = ['q_proj', 'k_proj', 'v_proj']  # 'out_proj', 'fc_in', 'fc_out', 'wte']

        peft_config = LoraConfig(**kwargs)
        optim = 'paged_adamw_32bit'
    else:
        peft_config = None

    if peft_config is not None:
        model.add_adapter(peft_config)

    return model, optim


def get_fromage_adapter(data_dim, hidden_dim, n_layers, final_dim, device):
    return FeatureExtractor(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        final_dim=final_dim).to(device)
