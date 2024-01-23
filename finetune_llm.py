"""Traiiin and Fine-tune TxGNN model."""
from absl import app
from absl import flags

import math
import numpy as np
import os
import torch

from pynvml import *
from scipy.stats import entropy
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from datasets import load_dataset
import evaluate
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    pipeline,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    PromptEncoderConfig,
)
from utils import load_txgnn_dataset


os.environ["WANDB_PROJECT"]="TxGNNv2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_PATH = '/n/home06/jschwarz/data/TxGNN/'

FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_integer('n_epochs', 5, 'Number of epochs.', lower_bound=1)
flags.DEFINE_integer('n_max_steps', 10, 'Maximum number of training steps.', lower_bound=1)
flags.DEFINE_integer('batch_size', 24, 'Finetuning Batch size.', lower_bound=1)
flags.DEFINE_integer('eval_batch_size', 24, 'Eval Batch size.', lower_bound=1)

# Model
flags.DEFINE_enum('model', 'distilbert', ['distilbert', 'llama2_7b', 'llama2_13b'], 'Model.')
flags.DEFINE_enum('finetune_type', 'full', ['full', 'lora'], 'Finetunting type.')
flags.DEFINE_boolean('lora_apply_everywhere', True, 'Whether to apply lora everywhere.')

# Misc
flags.DEFINE_boolean('wandb_track', False, 'Use WandB')
flags.DEFINE_enum('dataset', 'txgnn_did', ['guanaco', 'imdb', 'txgnn_did', 'txgnn_dod', 'txgnn_dcd', 'txgnn_drid', 'txgnn_drod', 'txgnn_drcd'], 'Dataset type.')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_string('checkpoint', './checkpoints/model_ckpt', 'Checkpoint location.')
flags.DEFINE_integer('valid_every', 250, 'Validation every #steps.', lower_bound=1)
flags.DEFINE_string('data_path', './data', 'Data location.')
flags.DEFINE_string('exp_name', 'debug', 'Experiment name.')

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

DD_TYPES = {
    'did': 'drug_indication_disease', 
    'dod': 'drug_off-label use_disease',
    'dcd': 'drug_contraindication_disease',
    'drid': 'disease_rev_indication_drug',
    'drod': 'disease_rev_off-label use_drug',
    'drcd': 'disease_rev_contraindication_drug',
}

accuracy = evaluate.load("accuracy")




def _get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    prediction_probs = softmax(logits, axis=1)
    predictions = np.argmax(logits, axis=1)

    relevant_prediction_probs = 1.0 - prediction_probs[
        range(labels.shape[0]), labels.astype(np.int32).flatten()][:, np.newaxis]

    metric_dict = accuracy.compute(predictions=predictions, references=labels)
    metric_dict['auroc'] = roc_auc_score(labels, relevant_prediction_probs)
    metric_dict['auprc'] = average_precision_score(labels, relevant_prediction_probs)
    return metric_dict



class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main(argv):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")

    #if FLAGS.wandb_track:
    #    import wandb
    #    wandb.init(project='TxGNNv2', name='{}_finetune ({})'.format(FLAGS.dataset, FLAGS.model))

    # Load relevant tokenizer
    if 'distilbert' == FLAGS.model:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    elif  'llama2_7b' == FLAGS.model:
        tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    elif  'llama2_13b' == FLAGS.model:
        tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    def _preprocess_function(examples):
        max_length = 20  # Length of bert
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)

    if 'imdb' in FLAGS.dataset:
        imdb = load_dataset("imdb")
        tokenized_dataset = imdb.map(_preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        task_type = "SEQ_CLS"
    elif 'guanaco' in FLAGS.dataset:
        dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
        task_type = "CAUSAL_LM"
    elif 'txgnn_d' in FLAGS.dataset:
        txgnn = load_txgnn_dataset(FLAGS.dataset.split('_')[1])
        tokenized_dataset = txgnn.map(_preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        task_type = "SEQ_CLS"

    # Load the entire model on the GPU 0
    device_map = {"": 0}
    use_bf16 = False
    if 'distilbert' == FLAGS.model:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            device_map=device_map,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
    elif 'llama2' in FLAGS.model:
        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, "float16")
        bf16 = False
        use_4bit = True
        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerating training with bf16=True")
                print("=" * 80)
                bf16 = True

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        if FLAGS.model == 'llama2_7b':
            model_type = "NousResearch/Llama-2-7b-hf"
        elif  FLAGS.model == 'llama2_13b':
            import pdb; pdb.set_trace()
            model_type = "NousResearch/Llama-2-13b-hf"

        if 'SEQ_CLS' == task_type:

            model = AutoModelForSequenceClassification.from_pretrained(
                model_type,
                quantization_config=bnb_config,
                device_map=device_map,
                num_labels=2,
                id2label=id2label,
                label2id=label2id,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_type,
                quantization_config=bnb_config,
                device_map=device_map
            )

            # Make predictions
            pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
            prompt = txgnn['test'][0]['text']
            result = pipe(f"<s>[INST] {prompt} [/INST]")
            print(result[0]['generated_text'])

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        use_bf16 = True

    print("=" * 80)
    print("Model loaded. " + _get_gpu_utilization())
    print("=" * 80)

    optim = 'adamw_torch'
    if 'lora' in FLAGS.finetune_type:

        kwargs = {
            'task_type': task_type,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'r': 64,
            'bias': "none",
            'modules_to_save': ["score"],
        }
        if not FLAGS.lora_apply_everywhere:
            kwargs['target_modules'] = ["q_proj", "k_proj", "v_proj"]  # "out_proj", "fc_in", "fc_out", "wte"]

        peft_config = LoraConfig(**kwargs)
        optim = 'paged_adamw_32bit'
    elif 'full' in FLAGS.finetune_type:
        peft_config = None

    if peft_config is not None:
        model.add_adapter(peft_config)

    # Set training parameters
    training_args = TrainingArguments(
	output_dir="~/logs/huggingface",
	num_train_epochs=FLAGS.n_epochs,
	#max_steps=FLAGS.n_max_steps,
	per_device_train_batch_size=FLAGS.batch_size,
	per_device_eval_batch_size=FLAGS.eval_batch_size,
        # Optimization settings
	learning_rate=2e-5,
	weight_decay=0.01,
	max_grad_norm=0.3,
	lr_scheduler_type='cosine',
	warmup_ratio=0.03,
        # Logging & Validation settings
        logging_steps=25,
	evaluation_strategy="steps",
        eval_steps=200,
	save_strategy="epoch",
	load_best_model_at_end=False,
        # Efficiency settings
	fp16=False,
	bf16=use_bf16,
        gradient_checkpointing=False,
	gradient_accumulation_steps=1,
	optim=optim,
        report_to="wandb" if FLAGS.wandb_track else "none",
        run_name='{}_finetune ({})'.format(FLAGS.dataset, FLAGS.model),
    )

    # Set supervised fine-tuning parameters
    trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_dataset["train"],
	eval_dataset=tokenized_dataset["valid"],
	tokenizer=tokenizer,
	data_collator=data_collator,
	compute_metrics=_compute_metrics,
    )
    # Train model
    print('Training')
    trainer.train()

    # Test set evaluation
    print('Test set evaluation')
    print(trainer.evaluate(tokenized_dataset['test']))

    # Make predictions
    if 'SEQ_CLS' == task_type:
        pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    elif 'SEQ_GEN' == task_type:
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

        # Make predictions
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        prompt = txgnn['test'][0]['text']
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])


if __name__ == '__main__':
  app.run(main)
