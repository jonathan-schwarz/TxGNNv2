"""Train and Fine-tune TxGNN model."""
from absl import app
from absl import flags

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
from train_utils import _CONFIG_EXCLUDE_KEYS

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_integer('n_epochs', 1, 'Number of epochs.', lower_bound=1)
flags.DEFINE_integer('n_max_steps', -1, 'Maximum number of training steps.', lower_bound=-1)
flags.DEFINE_integer('batch_size', 24, 'Finetuning Batch size.', lower_bound=1)


# Model
flags.DEFINE_boolean('use_feature_extractor', True, 'DKL')
flags.DEFINE_enum('model', 'dkl', ['distmult', 'mlp', 'dkl', 'mlp_llama2_7b', 'dkl_llama2_7b'], 'Model to use.')
flags.DEFINE_float('learning_rate', 0.01, 'LR')
flags.DEFINE_float('dkl_learning_rate_multiplier', 0.01, 'LR factor for GP hyperparameters')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for feature extractor')
flags.DEFINE_enum('scheduler_type', 'cosine_decay_with_warmup', ['cosine_decay', 'cosine_decay_with_warmup', 'multi_step_lr', 'valid_plateau'], 'LR Scheduler.')
flags.DEFINE_integer('grid_size', 64, 'DKL Grid size', lower_bound=2)
flags.DEFINE_integer('final_dim', 256, 'DKL Final Dim.', lower_bound=1)
flags.DEFINE_integer('hidden_dim', 256, 'DKL Hidden Dim.', lower_bound=1)
flags.DEFINE_integer('n_layers', 3, 'DKL Hidden Layers.', lower_bound=1)
# Only for LLMs
flags.DEFINE_boolean('use_fromage', True, 'Whether to use GNN features in LLM predictive model.')
flags.DEFINE_boolean('lora_apply_everywhere', True, 'Whether to apply lora everywhere.')
flags.DEFINE_enum('finetune_type', 'full', ['full', 'lora'], 'Finetunting type.')
# Only for DKL
flags.DEFINE_enum('strategy', 'grid_interpolation',
                  ['grid_interpolation', 'unwhitened'], 'Variational Strategy.')
flags.DEFINE_boolean('wandb_track', False, 'Whether to use wandb.')


# Misc
# Valid choices are ['did', 'dod', 'dcd', 'drid', 'drod', 'drcd']
flags.DEFINE_enum('dataset', 'txgnn_dod', ['txgnn_did', 'txgnn_dod', 'txgnn_dcd', 'txgnn_drid', 'txgnn_drod', 'txgnn_drcd'], 'Dataset type.')
flags.DEFINE_boolean('full_matrix_eval', True, 'Evaluate on full matrix')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_integer('valid_every', 25, 'Validation every #steps.', lower_bound=1)


def maybe_save(best_model_metrics, model_metrics, ckpt_path, model, optimizer, llm_state_dict=None, likelihood=None):
    for k, v in best_model_metrics.items():
        if 'auroc_auprc' == k:
            new_metric = model_metrics['valid_auroc'] * model_metrics['valid_auprc']
            do_save = new_metric > v
        else:
            continue

        if do_save:
            print("Saving new best model with metric '{}'. New: {:.3f} Old: {:.3f}".format(k, new_metric, v))
            save_model(ckpt_path, k, model, optimizer, llm_state_dict, likelihood)
            best_model_metrics[k] = new_metric

    return best_model_metrics


def save_model(path, metric_name, model, optimizer, llm_state_dict=None, likelihood=None):

    print('Saving checkpoint to {}'.format(path))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # TODO(schwarzjn): Save config
    ckpt_file_name = 'best_' + metric_name + '_model.pt'
    save_dict = {
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
    }

    if llm_state_dict is not None:
        save_dict['llm_state_dict'] = llm_state_dict
    if likelihood is not None:
        save_dict['likelihood_state_dict'] = likelihood.state_dict()

    torch.save(save_dict, os.path.join(path, ckpt_file_name))


def main(argv):
    # Fix random seed (TODO(schwarzjn): Also for DataLoader)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")

    ckpt_path = './checkpoints/finetune/{}_finetune_{}/model_ckpt_{}'.format(
        FLAGS.dataset, FLAGS.model, str(datetime.datetime.now()))

    if FLAGS.wandb_track:
        config = {v: getattr(FLAGS, v) for v in dir(FLAGS)}
        wandb.init(project='TxGNNv2', name='{}_finetune ({})'.format(FLAGS.dataset, FLAGS.model),
                   config=config, config_exclude_keys=_CONFIG_EXCLUDE_KEYS)

    # Load data from pretrained GNN
    dataset_type = 'embedding_text' if 'llama' in FLAGS.model else 'embedding'
    (train_loader, valid_loader, test_loader,
      num_train_points, data_dim, num_classes, inducing_x, tokenizer) = load_txgnn_dataset(
        FLAGS.dataset, dataset_type, FLAGS.model, FLAGS.batch_size, device
    )

    _assemble_batch = functools.partial(assemble_batch,
                                        model_type=FLAGS.model,
                                        use_fromage=FLAGS.use_fromage,
                                        device=device)

    # Build model
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
    else:
        llm = None


    # Predictive model
    model, likelihood = get_model(
        FLAGS.model, data_dim, num_classes,
        FLAGS.hidden_dim, FLAGS.n_layers, FLAGS.final_dim,
        FLAGS.strategy, inducing_x, device, FLAGS.use_feature_extractor)

    # Go into train mode
    model.train()
    if likelihood is not None:
        likelihood.train()
    if llm is not None:
        llm.train()

     # Optimizer & LR scheduler
    llm_state_dict = None
    params = []
    if 'llama2_7b' in FLAGS.model:
        llm_state_dict = get_trainable_parameters(llm)

        # Needs special handling so we only save PEFT parameters
        params.append({'params': [v for k, v in llm_state_dict.items()], 'weight_decay': FLAGS.weight_decay})
        if FLAGS.use_fromage:
            params.append({'params': fromage_adapter.parameters()})

    if 'dkl' in FLAGS.model:
        params.append({'params': model.gp_layer.hyperparameters(),
                       'lr': FLAGS.learning_rate * FLAGS.dkl_learning_rate_multiplier})
        params.append({'params': model.gp_layer.variational_parameters()})
        params.append({'params': likelihood.parameters()})

        if FLAGS.use_feature_extractor:
            params.append({'params': model.feature_extractor.parameters(), 'weight_decay': FLAGS.weight_decay})
    else:
        llm_state_dict = None
        params.append({'params': model.parameters()})

    # TODO(schwarzjn): Enabled paged optimizer
    optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)
    scheduler = load_scheduler(FLAGS.scheduler_type, optimizer, FLAGS.n_epochs, 
                               num_train_points // FLAGS.batch_size)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    if 'dkl' in FLAGS.model:
        loss_fn = gpytorch.mlls.VariationalELBO(
                likelihood, model.gp_layer, num_data=num_train_points)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_model_metrics = {
        'acc': 0.0,
        'auroc_auprc': -np.inf,
        'loss': np.inf,
    }

    step = 0
    with gpytorch.settings.num_likelihood_samples(1):
        for i in range(FLAGS.n_epochs):
            j = 0

            for batch in train_loader:
                optimizer.zero_grad()

                model_input, labels = _assemble_batch(batch)

                if 'dkl' == FLAGS.model:
                    # Get predictive output
                    output = model(**model_input)
                    loss = -loss_fn(output, labels[:, 0])

                    # Probability of predicting class 1
                    pred_prob = likelihood(output).probs.mean(0)[:, 1]
                elif 'llama2_7b' in FLAGS.model:
        
                    if FLAGS.use_fromage:
                        # Embedding each drug / disease feature separately
                        # TODO(schwarzjn): Should we process drug/disease embedding separately?
                        fromage_features = fromage_adapter(
                            model_input['gnn_embeddings'].reshape([FLAGS.batch_size*2, gnn_data_dim // 2])
                        ).reshape([FLAGS.batch_size, 2, data_dim])
                        model_input['gnn_embeddings'] = fromage_features

                    # Apply LLM
                    llm_output = llm(**model_input).to(torch.float32)
                    if 'mlp' in FLAGS.model:
                        # Apply Linear layer
                        output = model(llm_output)
                        loss = loss_fn(output, labels)

                        # Probability of predicting class 1
                        pred_prob = torch.sigmoid(output)
                    else:
                        # Apply GP
                        output = model(llm_output)
                        loss = -loss_fn(output, labels[:, 0])

                        # Probability of predicting class 1
                        pred_prob = likelihood(output).probs.mean(0)[:, 1]
                else:
                    # Get predictive output
                    output = model(**model_input)
                    loss = loss_fn(output, labels)

                    # Probability of predicting class 1
                    pred_prob = torch.sigmoid(output)

                pred_label = pred_prob.ge(0.5).float()
                try:
                    # TODO(schwarzjn): Fix error when we have only positive/only negative examples
                    train_auroc = roc_auc_score(
                        labels.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                    train_auprc = average_precision_score(
                        labels.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                except:
                    pass

                # Train step
                loss.backward()
                optimizer.step()

                if 'valid_plateau' != FLAGS.scheduler_type:
                    scheduler.step()

                # Metrics
                train_correct = pred_label.eq(labels.view_as(pred_label)).cpu().sum()
                train_acc = train_correct / FLAGS.batch_size

                print('Epoch/Iter: {}/{} - Train Loss: {:.3f} - Acc: {:.3f} - AUROC: {:.3f} - AUPRC: {:.3f}'.format(
                    i + 1, j + 1, loss.item(), train_acc, train_auroc, train_auprc))

                if FLAGS.wandb_track:
                    wandb.log({
                        'train_loss': loss, 'train_acc': train_acc, 'train_lr': optimizer.param_groups[0]['lr'],
                        'train_auroc': train_auroc, 'train_auprc': train_auprc})

                if 0 == (j % FLAGS.valid_every):
                    valid_auprc = 0.0
                    valid_auroc = 0.0
                    valid_correct = 0
                    valid_loss = 0.0
                    n_batches = 0
                    valid_predictions = [] 
                    # This gives us 16 samples from the predictive distribution (for a GP)
                    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
                        for valid_batch in valid_loader:
                            model_input, labels = _assemble_batch(valid_batch)

                            if 'dkl' == FLAGS.model:
                                # Get predictive output
                                output = model(**model_input)
                                valid_loss += -loss_fn(output, labels[:, 0])
                                pred_prob = likelihood(output).probs.mean(0)[:, 1]
                            elif 'llama2_7b' in FLAGS.model:
                    
                                if FLAGS.use_fromage:
                                    # Embedding each drug / disease feature separately
                                    # TODO(schwarzjn): Should we process drug/disease embedding separately?
                                    fromage_features = fromage_adapter(
                                        model_input['gnn_embeddings'].reshape([FLAGS.batch_size*2, gnn_data_dim // 2])
                                    ).reshape([FLAGS.batch_size, 2, data_dim])
                                    model_input['gnn_embeddings'] = fromage_features

                                # Apply LLM
                                llm_output = llm(**model_input).to(torch.float32)
                                if 'mlp' in FLAGS.model:
                                    # Apply Linear output
                                    output = model(llm_output)
                                    valid_loss += loss_fn(output, labels)

                                    # Probability of predicting class 1
                                    pred_prob = torch.sigmoid(output)
                                else:
                                    # Apply GP
                                    output = model(llm_output)
                                    valid_loss += -loss_fn(output, labels[:, 0])

                                    # Probability of predicting class 1
                                    pred_prob = likelihood(output).probs.mean(0)[:, 1]
                            else:
                                # Get predictive output
                                output = model(**model_input)
                                valid_loss += loss_fn(output, labels)
                                pred_prob = torch.sigmoid(output)

                            pred_label = pred_prob.ge(0.5).float()
                            valid_predictions.append(
                                pred_prob.cpu().numpy())
                            try:
                                valid_auroc += roc_auc_score(
                                    labels.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                                valid_auprc += average_precision_score(
                                    labels.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                            except:
                                pass

                            valid_correct += pred_label.eq(labels.view_as(pred_label)).cpu().sum()
                            n_batches += 1

                    # Track Validation loss
                    valid_acc = 100. * valid_correct.item() / float(len(valid_loader.dataset))
                    valid_loss = (valid_loss  / float(n_batches)).item()
                    valid_auroc = (valid_auroc  / float(n_batches))
                    valid_auprc = (valid_auprc  / float(n_batches))
                    valid_predictions = np.concatenate(valid_predictions, axis=0)

                    print('Epoch/Iter: {}/{} - Valid Loss: {:.3f} - Acc: {}/{} ({:.3f}%) - AUROC: {:.3f} - AUPRC: {:.3f}'.format(
                        i + 1, j + 1, valid_loss, valid_correct.item(), len(valid_loader.dataset), valid_acc, valid_auroc, valid_auprc)
                    )

                    if 'valid_plateau' == FLAGS.scheduler_type:
                        scheduler.step(loss)

                    log_dict = {'valid_loss': valid_loss, 'valid_acc': valid_acc,
                                'valid_auroc': valid_auroc, 'valid_auprc': valid_auprc}

                    # Consider saving the best model
                    best_model_metrics = maybe_save(best_model_metrics, log_dict, ckpt_path,
                        model, optimizer, llm_state_dict, likelihood)
                    if FLAGS.wandb_track:
                        wandb.log(log_dict)

                j += 1
                if j >= FLAGS.n_max_steps and FLAGS.n_max_steps != -1:
                    break
                step += 1


    print(80 * '=')
    print('Loading best checkpoint for final evaluation')
    print(80 * '=')

    ckpt = torch.load(os.path.join(ckpt_path, 'best_auroc_auprc_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if llm_state_dict is not None:
        # Needs special handling since we only saved PEFT parameters
        llm.load_state_dict(
            construct_llm_state_dict(llm, ckpt['likelihood_state_dict']))
        llm.eval()
    if likelihood is not None:
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        likelihood.eval()

    test_auprc = 0.0
    test_auroc = 0.0
    test_correct = 0
    test_loss = 0.0
    n_batches = 0
    test_predictions = [] 
    test_labels = [] 
    # This gives us 16 samples from the predictive distribution (for a GP)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for test_batch in test_loader:
            model_input, labels = _assemble_batch(test_batch)

            if 'dkl' == FLAGS.model:
                # Get predictive output
                output = model(**model_input)
                test_loss += -loss_fn(output, labels[:, 0])
                pred_prob = likelihood(output).probs.mean(0)[:, 1]
            elif 'llama2_7b' in FLAGS.model:
    
                if FLAGS.use_fromage:
                    # Embedding each drug / disease feature separately
                    # TODO(schwarzjn): Should we process drug/disease embedding separately?
                    fromage_features = fromage_adapter(
                        model_input['gnn_embeddings'].reshape([FLAGS.batch_size*2, gnn_data_dim // 2])
                    ).reshape([FLAGS.batch_size, 2, data_dim])
                    model_input['gnn_embeddings'] = fromage_features

                # Apply LLM
                llm_output = llm(**model_input).to(torch.float32)
                if 'mlp' in FLAGS.model:
                    # Apply Linear output
                    output = model(llm_output)
                    test_loss += loss_fn(output, labels)

                    # Probability of predicting class 1
                    pred_prob = torch.sigmoid(output)
                else:
                    # Apply GP
                    output = model(llm_output)
                    test_loss += -loss_fn(output, labels[:, 0])

                    # Probability of predicting class 1
                    pred_prob = likelihood(output).probs.mean(0)[:, 1]
            else:
                # Get predictive output
                output = model(**model_input)
                test_loss += loss_fn(output, labels)
                pred_prob = torch.sigmoid(output)

            pred_label = pred_prob.ge(0.5).float()
            test_predictions.append(pred_prob.cpu().numpy())
            test_labels.append(pred_label.cpu().numpy())
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
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    print('{}. test loss: {:.3f} - acc: {}/{} ({:.3f}%) - auroc: {:.3f} - auprc: {:.3f}'.format(
        FLAGS.dataset, test_loss, test_correct.item(), len(test_loader.dataset), test_acc, test_auroc, test_auprc)
    )
    log_dict = {'test_loss': test_loss, 'test_acc': test_acc,
                'test_auroc': test_auroc, 'test_auprc': test_auprc}

    if FLAGS.wandb_track:
        wandb.log(log_dict)
        fpr, tpr, _ = roc_curve(test_labels, test_predictions)
        data = [[x, y] for (x, y) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns=["fpr", "tpr"])
        wandb.log({
            "roc_curve": wandb.plot.line(
                table, "fpr", "tpr", title="Receiver operating characteristic")
        })

        # Save all checkpoints to wandb
        wandb.save(ckpt_path + '/*')

        np.savez_compressed(
            os.path.join(ckpt_path, 'predictions'), 
            valid_predictions=valid_predictions,
            test_predictions=test_predictions,
        )
        wandb.save(os.path.join(ckpt_path, 'predictions.npz'))

        if FLAGS.full_matrix_eval:
            matrix_loader, num_matrix_points = load_txgnn_dataset_matrix(
                FLAGS.dataset, dataset_type, FLAGS.model, FLAGS.batch_size, device,
            )
            print('Evaluating on matrix set')
            matrix_predictions = [] 
            # This gives us 16 samples from the predictive distribution (for a GP)
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
                for batch in matrix_loader:
                    model_input = _assemble_batch(batch, return_labels=False)

                    if 'dkl' == FLAGS.model:
                        # Get predictive output
                        output = model(**model_input)
                        pred_prob = likelihood(output).probs.mean(0)[:, 1]
                    elif 'llama2_7b' in FLAGS.model:
            
                        if FLAGS.use_fromage:
                            # Embedding each drug / disease feature separately
                            # TODO(schwarzjn): Should we process drug/disease embedding separately?
                            fromage_features = fromage_adapter(
                                model_input['gnn_embeddings'].reshape([FLAGS.batch_size*2, gnn_data_dim // 2])
                            ).reshape([FLAGS.batch_size, 2, data_dim])
                            model_input['gnn_embeddings'] = fromage_features

                        # Apply LLM
                        llm_output = llm(**model_input).to(torch.float32)
                        if 'mlp' in FLAGS.model:
                            # Apply Linear output
                            output = model(llm_output)

                            # Probability of predicting class 1
                            pred_prob = torch.sigmoid(output)
                        else:
                            # Apply GP
                            output = model(llm_output)

                            # Probability of predicting class 1
                            pred_prob = likelihood(output).probs.mean(0)[:, 1]
                    else:
                        # Get predictive output
                        output = model(**model_input)
                        pred_prob = torch.sigmoid(output)

                    matrix_predictions.append(pred_prob.cpu().numpy())

                np.savez_compressed(
                    os.path.join(ckpt_path, 'matrix_predictions'), 
                    matrix_predictions=np.concatenate(matrix_predictions, axis=0),
                )
                wandb.save(os.path.join(ckpt_path, 'matrix_predictions.npz'))


if __name__ == '__main__':
  app.run(main)
