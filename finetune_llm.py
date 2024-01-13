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

from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PromptEncoderConfig,
)

DATA_PATH = '/n/home06/jschwarz/data/TxGNN/'

FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_integer('n_epochs', 10, 'Number of epochs.', lower_bound=1)
flags.DEFINE_integer('batch_size', 32, 'Finetuning Batch size.', lower_bound=1)

# Model
flags.DEFINE_enum('model', 'llama2_7b', ['bert', 'llama2_7b'], 'Model.')
flags.DEFINE_enum('finetune_type', 'lora', ['full', 'lora', 'qlora', 'p'], 'Finetunting type.')

# Misc
flags.DEFINE_boolean('wandb_track', True, 'Use WandB')
# Valid choices are ['did', 'dod', 'dcd', 'drid', 'drod', 'drcd']
flags.DEFINE_string('dataset', 'txgnn_did', 'Dataset.')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_string('checkpoint', './checkpoints/model_ckpt', 'Checkpoint location.')
flags.DEFINE_integer('valid_every', 250, 'Validation every #steps.', lower_bound=1)
flags.DEFINE_string('data_path', './data', 'Data location.')
flags.DEFINE_string('exp_name', 'debug', 'Experiment name.')

# bitsandbytes parameters
gradient_accumulation_steps = 1  # Number of update steps to accumulate the gradients for
gradient_checkpointing = True  # Enable gradient checkpointing
max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)
learning_rate = 2e-4  # Initial learning rate (AdamW optimizer)
weight_decay = 0.001  # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"  # Optimizer to use
lr_scheduler_type = "cosine"  # Learning rate schedule
warmup_ratio = 0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# SFT parameters
max_seq_length = None
packing = False  # Pack multiple short examples in the same input sequence to increase efficiency

# Load the entire model on the GPU 0
device_map = {"": 0}


DD_TYPES = {
    'did': 'drug_indication_disease', 
    'dod': 'drug_off-label use_disease',
    'dcd': 'drug_contraindication_disease',
    'drid': 'disease_rev_indication_drug',
    'drod': 'disease_rev_off-label use_drug',
    'drcd': 'disease_rev_contraindication_drug',
}


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


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


def main(argv):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")


    if FLAGS.wandb_track:
        import wandb
        wandb.init(project='TxGNNv2', name='{}_finetune ({})'.format(FLAGS.dataset, FLAGS.model))

    # Load relevant tokenizer
    max_length = 49  # Length of bert
    if 'bert' == FLAGS.model:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif  'llama2_7b' == FLAGS.model:
        tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    if 'txgnn_d' in FLAGS.dataset:
        path = os.path.join(
            DATA_PATH, 'pretrained_mine/complex_disease/separate/{}.npz')

        data = np.load(path.format(
            DD_TYPES[FLAGS.dataset.split('_')[1]]))

        train_text = np.concatenate(
            [data['train_u_names'], data['train_v_names']], axis=1)
        train_text = ['Does {} treat {}?'.format(train_text[i][0], train_text[i][1]) for i in range(train_text.shape[0])]
        train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=max_length)
        train_y = data['train_labels'][:, 0].astype(np.int32).tolist()

        valid_text = np.concatenate(
            [data['valid_u_names'], data['valid_v_names']], axis=1)
        valid_text = ['Does {} treat {}?'.format(valid_text[i][0], valid_text[i][1]) for i in range(valid_text.shape[0])]
        valid_encodings = tokenizer(valid_text, truncation=True, padding=True, max_length=max_length)
        valid_y = data['valid_labels'][:, 0].astype(np.int32).tolist()

        test_text = np.concatenate(
            [data['test_u_names'], data['test_v_names']], axis=1)
        test_text = ['Does {} treat {}?'.format(test_text[i][0], test_text[i][1]) for i in range(test_text.shape[0])]
        test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=max_length)
        test_y = data['test_labels'][:, 0].astype(np.int32).tolist()

    train_set = Dataset(train_encodings, train_y)
    valid_set = Dataset(valid_encodings, valid_y)
    test_set = Dataset(test_encodings, test_y)

    num_train_points = len(train_text)
    num_valid_points = len(valid_text)
    num_test_points = len(test_text)
    inducing_x = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=FLAGS.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=FLAGS.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=FLAGS.batch_size, shuffle=False)

    if 'bert' == FLAGS.model:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            device_map=device_map
        )
        model.to(device)
    elif 'llama2_7b' == FLAGS.model:
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
        model = AutoModelForSequenceClassification.from_pretrained(
            "NousResearch/Llama-2-7b-chat-hf",
            quantization_config=bnb_config,
            device_map=device_map,
            return_dict=True
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

    print("=" * 80)
    print("Model loaded. " + get_gpu_utilization())
    print("=" * 80)

    if 'lora' in FLAGS.finetune_type:
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    elif 'p' in FLAGS.finetune_type:
        # Load LoRA configuration
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=20,
            encoder_hidden_size=128)
        model = get_peft_model(model, peft_config)
    elif 'full' in FLAGS.finetune_type:
        pass

    model.print_trainable_parameters()

    # Go into train mode
    model.train()

    # Optimizer, Criterion & LR scheduler   
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=2)

    for i in range(FLAGS.n_epochs):
        j = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Get predictive output
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            pred_label = output.logits.argmax(-1)

            # Train step
            loss.backward()
            # loss.backward()
            optimizer.step()

            # Metrics
            train_correct = pred_label.eq(labels.view_as(pred_label)).cpu().sum()
            train_acc = 100. * train_correct.item() / FLAGS.batch_size

            print('Epoch/Iter: {}/{} - Train Loss: {:.3f} - Acc: {:.3f}'.format(
                    i + 1, j + 1, loss.item(), train_acc))

            if FLAGS.wandb_track:
                wandb.log({
                    'train_loss': loss, 'train_acc': train_acc, 'train_lr': optimizer.param_groups[0]['lr']})

            # if 0 == (j % FLAGS.valid_every):
                valid_correct = 0
                valid_loss = 0.0
                n_batches = 0
                with torch.no_grad():
                    for batch in train_loader:
                        if torch.cuda.is_available():
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)

                        # Get predictive output
                        output = model(input_ids, attention_mask=attention_mask, labels=labels)
                        valid_loss += output.loss
                        pred_label = output.logits.argmax(-1)

                        valid_correct += pred_label.eq(labels.view_as(pred_label)).cpu().sum()
                        n_batches += 1

                # Track Validation loss
                valid_acc = 100. * valid_correct.item() / float(32 * n_batches)
                valid_loss = (valid_loss  / float(n_batches)).item()

                print('Epoch/Iter: {}/{} - Valid Loss: {:.3f} - Acc: {}/{} ({:.3f}%)'.format(
                    i + 1, j + 1, valid_loss, valid_correct.item(), 32 * n_batches, valid_acc)
                )

                wandb_log_dict = {'valid_loss': valid_loss, 'valid_acc': valid_acc}

                if FLAGS.wandb_track:
                    wandb.log(wandb_log_dict)

                scheduler.step(valid_loss)

            j += 1



    # Try the model
    prompt = train_text[np.randint(0, num_train_points)]
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


if __name__ == '__main__':
  app.run(main)
