"""Zero-shot evaluation of language models."""
from absl import app
from absl import flags

import datetime
import numpy as np
import os
import random
import pandas as pd
import torch
import transformers
import wandb

from data_utils import *
from models.llm_models import *
from train_utils import *

FLAGS = flags.FLAGS

# Model
flags.DEFINE_integer('batch_size', 48, 'Finetuning Batch size.', lower_bound=1)
flags.DEFINE_enum('model', 'llm_mlp_llama3_8b', ['llm_mlp_gemma_7b', 'llm_mlp_llama2_7b',
                                                 'llm_mlp_llama3_8b', 'llm_mlp_mistral_7b'], 'Model to use.')
flags.DEFINE_enum('dataset', 'txgnn_did', ['txgnn_did', 'txgnn_dod', 'txgnn_dcd', 'txgnn_drid', 'txgnn_drod', 'txgnn_drcd'], 'Dataset type.')
flags.DEFINE_integer('max_new_length', 50, 'Maximum generation length.', lower_bound=1)
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_boolean('wandb_track', True, 'Whether to use wandb.')

flags.DEFINE_enum('chat_model', 'llm_mlp_llama2_70b', ['llm_mlp_llama2_13b', 'llm_mlp_llama2_70b'], 'Chat Model to use.')
flags.DEFINE_integer('chat_max_new_length', 1, 'Maximum generation length for chat model.', lower_bound=1)


def to_label(x):
    if 'yes' == x:
        return 1.0
    elif 'sure' == x:
        return 1.0
    elif 'no' == x:
        return 0.0
    else:
        print('Failed to assign label to {}, defaulting to zero'.format(x))
        return 0.0


def main(argv):
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")

    ckpt_path = '/n/holystore01/LABS/mzitnik_lab/Users/jschwarz/TxGNNv2/checkpoints/zero_shot_temporal_eval/{}_eval_({},{})/model_ckpt_{}'.format(
        FLAGS.dataset, FLAGS.model, FLAGS.chat_model, str(datetime.datetime.now()))

    if FLAGS.wandb_track:
        config = {v: getattr(FLAGS, v) for v in dir(FLAGS)}
        wandb.init(project='TxGNNv2', name='{}_zero_shot_eval ({}, {})'.format(FLAGS.dataset, FLAGS.model, FLAGS.chat_model),
                   config=config, config_exclude_keys=CONFIG_EXCLUDE_KEYS)

    # Load data from pretrained GNN
    task_type = 'TXT_GEN'
    (_, _, test_loader, _, _, _, _, tokenizer) = load_txgnn_dataset(
        FLAGS.dataset, 'embedding_text', 'v1', FLAGS.model, FLAGS.batch_size, device, task_type
    )

    # Auto-Regressive model
    llm, _, _ = get_llm(FLAGS.model, task_type, tokenizer, use_4bit=True, use_double_quant=False)
    llm.eval()

    all_queries = []
    all_answers = []
    all_labels = []
    for batch in test_loader:
        # Decode query into natural language
        all_queries += tokenizer.batch_decode(batch[1], skip_special_tokens=True)

        # Predictions
        output_ids = llm.generate(batch[1].to(device), attention_mask=batch[2].to(device),
                                  max_new_tokens=FLAGS.max_new_length, num_beams=5)

        # Decode answer into natural language
        all_answers += tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_labels.append(batch[-1].cpu().numpy())

     assert len(all_answers) == 960

    all_labels = np.concatenate(all_labels, axis=0)[:, 0]
    # Remove original question and whitespace
    all_answers = [a.replace(q, '').strip() for a, q in zip(all_answers, all_queries)]

    # Empty VRAM for Chat model
    del llm
    import gc
    gc.collect()
    gc.collect()

    # Chat evaluation model
    chat_tokenizer = get_tokenizer(FLAGS.chat_model, 'TXT_GEN')
    llm_2, _, _ = get_llm(FLAGS.chat_model, task_type, chat_tokenizer, use_4bit=True, use_double_quant=False,
                          device_map="auto")
    llm_2.eval()

    prompt_template='''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. I will share a medical question and doctor's answer.
    Analyze both question and anwer and tell me if the doctor's answer to the question is affirmative.
    I'm not interested in your personal opinion, only the response of the doctor.

    Please only respond with 'yes', 'no'.
    <</SYS>>
    Original query: {}
    Doctor's answer: {}.[/INST]
    '''

    # Evaluate responses
    all_chat_resonses = []
    for query, answer in zip(all_queries, all_answers):
        prompt = prompt_template.format(query, answer)
        tokenized_prompt = chat_tokenizer(prompt, return_tensors="pt").to(llm_2.device)
        output_ids = llm_2.generate(**tokenized_prompt,
                                    max_new_tokens=FLAGS.chat_max_new_length,
                                    do_sample=False, early_stopping=False)

        # Decode into natural language
        chat_response = chat_tokenizer.batch_decode(output_ids[:, -1:], skip_special_tokens=True)
        all_chat_resonses.append(chat_response[0].lower())

    predictions = [to_label(r) for r in all_chat_resonses]
    acc = (np.array(predictions) == np.array(all_labels)).sum()/all_labels.shape[0]
    print('Evaluation done. Accuracy: {:.3f}'.format(acc))

    df = pd.DataFrame({'query': all_queries,
                       'answer': all_answers,
                       'chat_response': all_chat_resonses,
                       'prediction': predictions,
                       'label': all_labels.tolist()})

    if FLAGS.wandb_track:
        wandb.log({'accuracy': acc})

        os.makedirs(ckpt_path)
        full_path = os.path.join(ckpt_path, 'prediction_data.csv')
        # Save data
        df.to_csv(full_path)
        wandb.save(full_path)


if __name__ == '__main__':
  app.run(main)
