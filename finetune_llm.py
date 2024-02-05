"""Traiiin and Fine-tune TxGNN model."""
from absl import app
from absl import flags

import datetime
import math
import numpy as np
import os
import torch
import wandb

from datasets import load_dataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
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
from pynvml import *

from finetune_models.llm_models import *
from data_utils import load_txgnn_dataset_text

os.environ["WANDB_PROJECT"] = "TxGNNv2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_integer('n_epochs', 1, 'Number of epochs.', lower_bound=1)
flags.DEFINE_integer('n_max_steps', -1, 'Maximum number of training steps.', lower_bound=-1)
flags.DEFINE_integer('batch_size', 24, 'Finetuning Batch size.', lower_bound=1)
flags.DEFINE_integer('eval_batch_size', 24, 'Eval Batch size.', lower_bound=1)

# Model
flags.DEFINE_enum('best_model_metric', 'loss', ['loss', 'auroc_auprc'], 'What metric to use for early stopping.')
flags.DEFINE_enum('model', 'llama2_7b', ['distilbert', 'llama2_7b', 'llama2_13b'], 'Model.')
flags.DEFINE_enum('finetune_type', 'full', ['full', 'lora'], 'Finetunting type.')
flags.DEFINE_boolean('lora_apply_everywhere', True, 'Whether to apply lora everywhere.')

# Misc
flags.DEFINE_boolean('wandb_track', False, 'Use WandB')
flags.DEFINE_enum('dataset', 'txgnn_did', ['txgnn_did', 'txgnn_dod', 'txgnn_dcd', 'txgnn_drid', 'txgnn_drod', 'txgnn_drcd'], 'Dataset type.')
flags.DEFINE_boolean('dataset_use_v2', False, 'Use v2 Dataset (more negatives).')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_integer('valid_every', 250, 'Validation every #steps.', lower_bound=1)
flags.DEFINE_string('data_path', './data', 'Data location.')
flags.DEFINE_string('exp_name', 'debug', 'Experiment name.')


def main(argv):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")

    ckpt_path = './checkpoints/finetune_llm/{}_finetune_{}/model_ckpt_{}'.format(
        FLAGS.dataset, FLAGS.model, str(datetime.datetime.now()))
    print('Saving results to {}'.format(ckpt_path))

    tokenizer = get_tokenizer(FLAGS.model)
    tokenized_dataset, data_collator, task_type = load_txgnn_dataset_text(
        FLAGS.dataset, FLAGS.dataset_use_v2, tokenizer)

    model, use_bf16 = get_model(FLAGS.model, task_type, tokenizer)
    model, optim = get_peft(model, task_type, FLAGS.finetune_type, FLAGS.lora_apply_everywhere)

    # Set training parameters
    training_args = TrainingArguments(
	output_dir=ckpt_path,
	num_train_epochs=FLAGS.n_epochs,
	max_steps=FLAGS.n_max_steps,
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
        eval_steps=100,
	save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        metric_for_best_model=FLAGS.best_model_metric,
        greater_is_better=False if 'eval_loss' == FLAGS.best_model_metric else True,
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
	compute_metrics=compute_metrics,
    )
    # Train model
    print('Training')
    trainer.train()

    # Test set evaluation
    print('Test set evaluation')
    print(trainer.evaluate(tokenized_dataset['test'], metric_key_prefix='test'))

    # Save best model
    best_model_path = os.path.join(ckpt_path, FLAGS.model + '_{}'.format(FLAGS.dataset))
    trainer.model.save_pretrained(best_model_path)

    # Make predictions
    if 'SEQ_CLS' == task_type:
        pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    elif 'SEQ_GEN' == task_type:
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    if FLAGS.wandb_track:
        fpr, tpr, _ = roc_curve(test_pred_outputs.label_ids, torch.nn.Softmax()(torch.Tensor(test_pred_outputs.predictions))[:, 1].numpy())
        data = [[x, y] for (x, y) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns=["fpr", "tpr"])
        wandb.log({
            "roc_curve": wandb.plot.line(
                table, "fpr", "tpr", title="Receiver operating characteristic")
        })

        # Save all checkpoints to wandb
        wandb.save(best_model_path + '/*')

        train_pred_outputs = trainer.predict(tokenized_dataset['train'])
        valid_pred_outputs = trainer.predict(tokenized_dataset['valid'])
        test_pred_outputs = trainer.predict(tokenized_dataset['test'])

        np.savez_compressed(
            os.path.join(ckpt_path, 'predictions'), 
            train_predictions=torch.nn.Softmax()(torch.Tensor(train_pred_outputs.predictions))[:, 1].numpy(),
            valid_predictions=torch.nn.Softmax()(torch.Tensor(valid_pred_outputs.predictions))[:, 1].numpy(),
            test_predictions=torch.nn.Softmax()(torch.Tensor(test_pred_outputs.predictions))[:, 1].numpy(),
        )

        wandb.save(os.path.join(ckpt_path, 'predictions.npz'))


if __name__ == '__main__':
  app.run(main)
