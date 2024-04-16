"""Evaluate model in bandit-style setting."""
from absl import app
from absl import flags

import collections
import datetime
import functools
import gpytorch
import math
import numpy as np
import os
import random
import torch
import wandb

from data_utils import *
from finetune_models.models import *
from finetune_models.llm_models import *
from train_utils import *

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_integer('batch_size', 1, 'Return one disease at a time.', lower_bound=1)
flags.DEFINE_integer('eval_batch_size', 128, 'Return one disease at a time.', lower_bound=1)
flags.DEFINE_integer('n_epochs', 1, 'Number of epochs.', lower_bound=1)
flags.DEFINE_boolean('zero_shot', True, 'Whether to update model.')

# Model
flags.DEFINE_string('ckpt_path', '/n/home06/jschwarz/git/TxGNNv2/checkpoints/finetune/txgnn_did_finetune_distmult/model_ckpt_2024-04-15 09:26:53.767800/', 'Checkpoint')
flags.DEFINE_boolean('use_feature_extractor', True, 'DKL')
flags.DEFINE_enum('model', 'dkl', ['distmult', 'mlp', 'dkl', 'mlp_llama2_7b', 'dkl_llama2_7b', 'dkl_mixtral'], 'Model to use.')
flags.DEFINE_float('learning_rate', 0.01, 'LR')
flags.DEFINE_float('max_grad_norm', 10.0, 'Used for optional gradient clipping')
flags.DEFINE_float('dkl_learning_rate_multiplier', 0.01, 'LR factor for GP hyperparameters')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for feature extractor')
flags.DEFINE_enum('scheduler_type', 'cosine_decay_with_warmup', ['constant', 'cosine_decay', 'cosine_decay_with_warmup', 'multi_step_lr', 'valid_plateau'], 'LR Scheduler.')
flags.DEFINE_integer('grid_size', 64, 'DKL Grid size', lower_bound=2)
flags.DEFINE_integer('final_dim', 256, 'DKL Final Dim.', lower_bound=1)
flags.DEFINE_integer('hidden_dim', 256, 'DKL Hidden Dim.', lower_bound=1)
flags.DEFINE_integer('n_layers', 3, 'DKL Hidden Layers.', lower_bound=1)
# Only for LLMs
flags.DEFINE_boolean('use_fromage', True, 'Whether to use GNN features in LLM predictive model.')
flags.DEFINE_boolean('lora_apply_everywhere', True, 'Whether to apply lora everywhere.')
flags.DEFINE_enum('finetune_type', 'full', ['full', 'lora', 'none'], 'LLM Finetunting type. Disabled when `none`')
# Only for DKL
flags.DEFINE_enum('strategy', 'grid_interpolation',
                  ['grid_interpolation', 'unwhitened'], 'Variational Strategy.')
flags.DEFINE_boolean('wandb_track', False, 'Whether to use wandb.')


# Misc
# Valid choices are ['did', 'dod', 'dcd', 'drid', 'drod', 'drcd']
flags.DEFINE_enum('dataset', 'txgnn_did', ['txgnn_did'], 'Dataset type.')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_integer('valid_every', 25, 'Validation every #steps.', lower_bound=1)

# ROWS: True label (0: No Indication 1: Indication, 2: Contraindication, 3: Off-label use 4: Unknown)
# COLS: Predicted label (0: No Indication 1: Indication)
_REWARD_MATRIX  = np.array([[1, -1],
                            [-1, 3],
                            [2, -1],
                            [0, 1],
                            [0, 0]])


def get_reward(predicted_outcome, true_outcome):
    reward = _REWARD_MATRIX[
        true_outcome, predicted_outcome]
    
    return reward

def optimal_policy(labels):
    if 1.0 in labels:
        top_id = labels.index(1.0)
        predicted_outcome = 1 
    elif 2.0 in labels: 
        top_id = labels.index(2.0)
        predicted_outcome = 0
    elif 0.0 in labels:
        top_id = labels.index(0.0)
        predicted_outcome = 0 
    elif 3.0 in labels:
        top_id = labels.index(3.0)
        predicted_outcome = 1 
    elif 4.0 in labels:
        top_id = labels.index(4.0)
        predicted_outcome = 1 

    return top_id, predicted_outcome

def random_policy(labels):
    top_id = np.random.randint(0, len(labels))
    predicted_outcome = np.random.randint(0, 2)

    return top_id, predicted_outcome


def main(argv):
    # Fix random seed (TODO(schwarzjn): Also for DataLoader)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # TODO(schwarzjn): Remove once fixed
    assert FLAGS.zero_shot, 'Training not yet available'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")

    ckpt_path = FLAGS.ckpt_path


    if FLAGS.wandb_track:
        config = {v: getattr(FLAGS, v) for v in dir(FLAGS)}
        wandb.init(project='TxGNNv2', name='{}_bandit_eval ({})'.format(FLAGS.dataset, FLAGS.model),
                   config=config, config_exclude_keys=CONFIG_EXCLUDE_KEYS)

    # Load data from pretrained GNN (in format [diseases, drugs, features])
    _assemble_batch = functools.partial(assemble_batch,
                                        model_type=FLAGS.model,
                                        use_fromage=FLAGS.use_fromage,
                                        device=device)

    dataset_type = 'embedding_text' if 'llama' in FLAGS.model else 'embedding'
    # For test set evaluation (merely as a santiy check)
    (_, _, test_loader,
      num_train_points, data_dim, num_classes, inducing_x, tokenizer) = load_txgnn_dataset(
        FLAGS.dataset, dataset_type, FLAGS.model, FLAGS.eval_batch_size, device
    )

    # For full matrix evaluation
    (matrix_loader, num_matrix_points, data_dim, num_classes, tokenizer) = load_txgnn_dataset_matrix(
        FLAGS.dataset, dataset_type, FLAGS.model, 1, device, eval_bandit=True,
    )
    inducing_x = None

    # Build model
    fromage_adapter = None
    llm = None
    if 'llama' in FLAGS.model:
        # llama embedding dimension
        data_dim = 4096

        # Language Model
        llm, use_bf16 = get_llm(FLAGS.model, 'SEQ_CLS', tokenizer)
        llm, optim = get_peft(
            llm, 'SEQ_CLS', FLAGS.finetune_type, FLAGS.lora_apply_everywhere,
            use_final_layer=False)

        if FLAGS.use_fromage:
            # Adapter for GNN features
            gnn_data_dim = 1024
            fromage_adapter = get_fromage_adapter(
                gnn_data_dim // 2, FLAGS.hidden_dim, FLAGS.n_layers, data_dim, llm.device)


    # Predictive model
    model, likelihood = get_model(
        FLAGS.model, data_dim, num_classes,
        FLAGS.hidden_dim, FLAGS.n_layers, FLAGS.final_dim,
        FLAGS.strategy, inducing_x, device, FLAGS.use_feature_extractor)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    if 'dkl' in FLAGS.model:
        loss_fn = gpytorch.mlls.VariationalELBO(
                likelihood, model.gp_layer, num_data=num_train_points)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    ckpt = torch.load(os.path.join(ckpt_path, 'best_auroc_auprc_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if 'llama' in FLAGS.model:
        # Needs special handling since we only saved PEFT parameters
        llm.load_state_dict(
            construct_llm_state_dict(llm, ckpt['llm_state_dict']))
        llm.eval()
    if likelihood is not None:
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        likelihood.eval()

    test_auprc = 0.0
    test_auroc = 0.0
    test_correct = 0
    test_loss = 0.0
    n_batches = 0
    # This gives us 16 samples from the predictive distribution (for a GP)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for test_batch in test_loader:
            model_input, labels = _assemble_batch(test_batch)

            pred_prob, pred_label, loss = forward_pass(
                FLAGS.model, FLAGS.use_fromage, model, llm, fromage_adapter, likelihood,
                model_input, labels, loss_fn,
            )
            test_loss += loss

            try:
                test_auroc += roc_auc_score(labels.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                test_auprc += average_precision_score(labels.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
            except:
                pass
            correct_predictions = pred_label.eq(labels.view_as(pred_label)).cpu().numpy()
            test_correct += correct_predictions.sum()
            n_batches += 1

    # Track Test loss
    test_acc = 100. * test_correct / float(len(test_loader.dataset))
    test_loss = (test_loss  / float(n_batches)).item()
    test_auroc = (test_auroc  / float(n_batches))
    test_auprc = (test_auprc  / float(n_batches))

    print('{}. test loss: {:.3f} - acc: {}/{} ({:.3f}%) - auroc: {:.3f} - auprc: {:.3f}'.format(
        FLAGS.dataset, test_loss, test_correct.item(), len(test_loader.dataset), test_acc, test_auroc, test_auprc)
    )

    all_optimal_rewards = []
    all_real_rewards = []
    all_random_rewards = []
    all_true_outcomes = []

    i = 0
    for batch in matrix_loader:
        batch = [b[0] for b in batch]

        optimal_reward = 0
        labels = batch[-1].cpu().numpy()[:, 0].tolist()
        # Run optimal policy
        for j in range(NUM_DRUGS):
            top_id, predicted_outcome = optimal_policy(labels)
            optimal_reward += get_reward(predicted_outcome, int(labels[top_id]))
            # Pop predicted element
            if j < NUM_DRUGS - 1:
                labels = labels[:top_id] + labels[top_id+1:]

        random_reward = 0
        labels = batch[-1].cpu().numpy()[:, 0].tolist()
        # Run random policy
        for j in range(NUM_DRUGS):
            top_id, predicted_outcome = random_policy(labels)
            random_reward += get_reward(predicted_outcome, int(labels[top_id]))
            # Pop predicted element
            if j < NUM_DRUGS - 1:
                labels = labels[:top_id] + labels[top_id+1:]

        real_reward = 0
        true_outcomes = []

        labels = batch[-1].cpu().numpy()[:, 0].tolist()
        # Run Real policy
        for j in range(NUM_DRUGS):
            
            if j == 0 or not FLAGS.zero_shot:
                # Collect predictions for all drugs
                pred_probs = []
                pred_labels = []
                num_batches = len(labels) // FLAGS.eval_batch_size + int(len(labels) % FLAGS.eval_batch_size > 0)
                for k in range(num_batches):
                    mini_batch = [b[k*FLAGS.eval_batch_size:(k+1)*FLAGS.eval_batch_size] for b in batch]
                    model_input = _assemble_batch(mini_batch, return_labels=False)

                    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
                        output = forward_pass(
                            FLAGS.model, FLAGS.use_fromage, model, llm, fromage_adapter, likelihood,
                            model_input, return_loss=False,
                        )

                    pred_probs.append(output[0])
                    pred_labels.append(output[1])

                pred_prob = torch.concat(pred_probs, axis=0)
                pred_label = torch.concat(pred_labels, axis=0)

            top_id = torch.argmax(pred_prob)
            predicted_outcome = int(pred_label[top_id].item())
            true_outcome = int(labels[top_id])
            true_outcomes.append(true_outcome)
            real_reward += get_reward(predicted_outcome, true_outcome)
        
            # Pop predicted element
            if j < NUM_DRUGS - 1:
                labels = labels[:top_id] + labels[top_id+1:]
                new_batch = []
                for b in batch:
                    new_batch.append(torch.concat([b[:top_id], b[top_id+1:]]))
                pred_prob = torch.concat([pred_prob[:top_id], pred_prob[top_id+1:]])
                pred_label = torch.concat([pred_label[:top_id], pred_label[top_id+1:]])

            batch = new_batch

        print('Disease: {} Optimal reward: {} Real reward: {} Random reward: {}'.format(
            i, optimal_reward, real_reward, random_reward))
        all_optimal_rewards.append(optimal_reward)
        all_real_rewards.append(real_reward)
        all_random_rewards.append(random_reward)
        all_true_outcomes.append(true_outcomes)
        i += 1 

    if FLAGS.wandb_track:
        wandb.log({
            "optimal_reward_mean": np.mean(all_optimal_rewards),
            "optimal_reward_std": np.std(all_optimal_rewards),
            "real_reward_mean": np.mean(all_real_rewards),
            "real_reward_std": np.std(all_real_rewards),
            "random_reward_mean": np.mean(all_random_rewards),
            "random_reward_std": np.std(all_random_rewards),
        })

        np.savez_compressed(
            os.path.join(ckpt_path, 'bandit_eval_outcomes'),
            true_outcomes=np.array(all_true_outcomes),
            optimal_rewards=np.array(all_optimal_rewards),
            real_rewards=np.array(all_real_rewards),
            random_rewards=np.array(all_random_rewards),
        )
        wandb.save(os.path.join(ckpt_path, 'bandit_eval_outcomes.npz'))



if __name__ == '__main__':
  app.run(main)
