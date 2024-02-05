"""Train and Fine-tune TxGNN model."""
from absl import app
from absl import flags

import datetime
import gpytorch
import math
import numpy as np
import os
import torch
import wandb

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from finetune_models.models import *
from data_utils import load_txgnn_dataset_embedding

FLAGS = flags.FLAGS

# Training settings
flags.DEFINE_integer('n_epochs', 1, 'Number of epochs.', lower_bound=1)
flags.DEFINE_integer('batch_size', 128, 'Finetuning Batch size.', lower_bound=1)

# Model
flags.DEFINE_boolean('use_feature_extractor', True, 'DKL')
flags.DEFINE_enum('model', 'dkl', ['dkl', 'mlp', 'distmult'], 'Model.')
flags.DEFINE_enum('strategy', 'grid_interpolation',
                  ['grid_interpolation', 'unwhitened'], 'Variational Strategy.')
flags.DEFINE_float('learning_rate', 0.01, 'LR')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay for feature extractor')
flags.DEFINE_enum('scheduler', 'valid_plateau', ['train_plateau', 'multi_step_lr', 'valid_plateau'], 'LR Scheduler.')
flags.DEFINE_integer('grid_size', 64, 'DKL Grid size', lower_bound=2)
flags.DEFINE_integer('final_dim', 256, 'DKL Final Dim.', lower_bound=1)
flags.DEFINE_integer('hidden_dim', 256, 'DKL Hidden Dim.', lower_bound=1)
flags.DEFINE_integer('n_inducing', 100, 'Number of inducing points.', lower_bound=1)
flags.DEFINE_integer('n_layers', 3, 'DKL Hidden Layers.', lower_bound=1)


# Misc
flags.DEFINE_boolean('wandb_track', False, 'Use WandB')
# Valid choices are ['did', 'dod', 'dcd', 'drid', 'drod', 'drcd']
flags.DEFINE_enum('dataset', 'txgnn_dod', ['txgnn_did', 'txgnn_dod', 'txgnn_dcd', 'txgnn_drid', 'txgnn_drod', 'txgnn_drcd'], 'Dataset type.')
flags.DEFINE_boolean('dataset_use_v2', False, 'Use v2 Dataset (more negatives).')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_string('checkpoint', './checkpoints/finetune/model_ckpt', 'Checkpoint location.')
flags.DEFINE_integer('valid_every', 25, 'Validation every #steps.', lower_bound=1)
flags.DEFINE_string('data_path', './data', 'Data location.')


def maybe_save(best_model_metrics, model_metrics, ckpt_path, model, optimizer, likelihood):
    for k, v in best_model_metrics.items():
        if 'auroc_auprc' == k:
            new_metric = model_metrics['valid_auroc'] * model_metrics['valid_auprc']
            do_save = new_metric > v
        elif 'loss' == k:
            new_metric = model_metrics['valid_loss']
            do_save = new_metric < v
        elif 'acc' == k:
            new_metric = model_metrics['valid_acc']
            do_save = new_metric > v

        if do_save:
            print("Saving new best model with metric '{}'. New: {:.3f} Old: {:.3f}".format(k, new_metric, v))
            save_model(ckpt_path, k, model, optimizer, likelihood)
            best_model_metrics[k] = new_metric

    return best_model_metrics


def save_model(path, metric_name, model, optimizer, likelihood=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # TODO(schwarzjn): Save config
    ckpt_file_name = 'best_' + metric_name + '_model.pt'
    save_dict = {
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
    }
    if likelihood is not None:
        save_dict['likelihood_state_dict'] = likelihood.state_dict()

    torch.save(save_dict, os.path.join(path, ckpt_file_name))


def main(argv):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")

    ckpt_path = './checkpoints/finetune/{}_finetune_{}/model_ckpt_{}'.format(
        FLAGS.dataset, FLAGS.model, str(datetime.datetime.now()))

    if FLAGS.wandb_track:
        wandb.init(project='TxGNNv2', name='{}_finetune ({})'.format(FLAGS.dataset, FLAGS.model))

    (train_loader, valid_loader, test_loader,
      num_train_points, data_dim, num_classes, inducing_x) = load_txgnn_dataset_embedding(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.dataset_use_v2, device,
    )

    model, likelihood = get_model(
        FLAGS.model, data_dim, num_classes,
        FLAGS.hidden_dim, FLAGS.n_layers, FLAGS.final_dim,
        FLAGS.strategy, inducing_x, device)

    # Go into train mode
    model.train()
    if likelihood is not None:
        likelihood.train()

     # Optimizer & LR scheduler
    if 'dkl' == FLAGS.model:
        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
            {'params': model.gp_layer.hyperparameters(), 'lr': FLAGS.learning_rate * 0.01},
            {'params': model.gp_layer.variational_parameters()},
            {'params': likelihood.parameters()},
        ], lr=FLAGS.learning_rate)
    else:
        params = [v for v in model.parameters()]
        optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)

    if 'train_plateau' == FLAGS.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=25)
    elif 'valid_plateau' == FLAGS.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=5)
    elif 'multi_step_lr' == FLAGS.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[0.25 * FLAGS.n_epochs, 0.5 * FLAGS.n_epochs, 0.75 * FLAGS.n_epochs], gamma=0.1)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    if 'dkl' == FLAGS.model:
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model.gp_layer, num_data=num_train_points)
    else:
        if num_classes == 2:
            mll = torch.nn.BCEWithLogitsLoss()
        else:
            mll = torch.nn.CrossEntropyLoss()

    best_model_metrics = {
        'acc': 0.0,
        'auroc_auprc': -np.inf,
        'loss': np.inf,
    }

    with gpytorch.settings.num_likelihood_samples(1):
        for i in range(FLAGS.n_epochs):
            j = 0
            for train_x, train_y in train_loader:
                if torch.cuda.is_available():
                    train_x, train_y = train_x.cuda(), train_y.cuda()
                    # Make compatible with MLP
                    train_x = train_x.view(train_x.size(0), -1)

                optimizer.zero_grad()

                # Get predictive output
                output = model(train_x)

                if 'dkl' == FLAGS.model:
                    loss = -mll(output, train_y[:, 0])
                    # Probability of predicting class 1
                    pred_prob = likelihood(output).probs.mean(0)[:, 1]
                    pred_label = pred_prob.ge(0.5).float()
                else:
                    if num_classes == 2:
                        loss = mll(output, train_y)
                        # Probability of predicting class 1
                        pred_prob = torch.sigmoid(output)
                        pred_label = pred_prob.ge(0.5).float()
                    else:
                        loss = mll(output, train_y.long())
                        pred_label = torch.argmax(output, -1)

                train_auroc = roc_auc_score(
                    train_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                train_auprc = average_precision_score(
                    train_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())

                # Train step
                loss.backward()
                optimizer.step()

                if 'train_plateau' == FLAGS.scheduler:
                    scheduler.step(loss)

                # Metrics
                train_correct = pred_label.eq(train_y.view_as(pred_label)).cpu().sum()
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
                    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
                        for valid_x, valid_y in valid_loader:
                            if torch.cuda.is_available():
                                valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
                                # Make compatible with MLP
                                valid_x = valid_x.view(valid_x.size(0), -1)

                            # This gives us 16 samples from the predictive distribution (for a GP)
                            output = model(valid_x)
                            if 'dkl' == FLAGS.model:
                                valid_loss += -mll(output, valid_y[:, 0])
                                pred_prob = likelihood(output).probs.mean(0)[:, 1]
                                pred_label = pred_prob.ge(0.5).float()
                            else:
                                if num_classes == 2:
                                    valid_loss += mll(output, valid_y)
                                    pred_prob = torch.sigmoid(output)
                                    pred_label = pred_prob.ge(0.5).float()
                                else:
                                    valid_loss += mll(output, valid_y.long())
                                    pred_label = torch.argmax(output, -1)

                            valid_auroc += roc_auc_score(
                                valid_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                            valid_auprc += average_precision_score(
                                valid_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
                            valid_correct += pred_label.eq(valid_y.view_as(pred_label)).cpu().sum()
                            n_batches += 1

                    # Track Validation loss
                    valid_acc = 100. * valid_correct.item() / float(len(valid_loader.dataset))
                    valid_loss = (valid_loss  / float(n_batches)).item()
                    valid_auroc = (valid_auroc  / float(n_batches))
                    valid_auprc = (valid_auprc  / float(n_batches))

                    print('Epoch/Iter: {}/{} - Valid Loss: {:.3f} - Acc: {}/{} ({:.3f}%) - AUROC: {:.3f} - AUPRC: {:.3f}'.format(
                        i + 1, j + 1, valid_loss, valid_correct.item(), len(valid_loader.dataset), valid_acc, valid_auroc, valid_auprc)
                    )

                    if 'valid_plateau' == FLAGS.scheduler:
                        scheduler.step(loss)

                    # Consider saving the best model

                    log_dict = {'valid_loss': valid_loss, 'valid_acc': valid_acc,
                                'valid_auroc': valid_auroc, 'valid_auprc': valid_auprc}
                    best_model_metrics = maybe_save(best_model_metrics, log_dict,
                                                    ckpt_path, model, optimizer, likelihood)
                    if FLAGS.wandb_track:
                        wandb.log(log_dict)

                j += 1

        if 'multi_step_lr' == FLAGS.scheduler:
            scheduler.step()


    print(80 * '=')
    print('Loading best checkpoint for final evaluation')
    print(80 * '=')
    ckpt = torch.load(os.path.join(ckpt_path, 'best_auroc_auprc_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    model.eval()
    if likelihood is not None:
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])
        likelihood.eval()

    test_auprc = 0.0
    test_auroc = 0.0
    test_correct = 0
    test_loss = 0.0
    n_batches = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                test_x, test_y = test_x.cuda(), test_y.cuda()
                # Make compatible with MLP
                test_x = test_x.view(test_x.size(0), -1)

            # This gives us 16 samples from the predictive distribution (for a GP)
            output = model(test_x)

            if 'dkl' == FLAGS.model:
                test_loss += -mll(output, test_y[:, 0])
                pred_prob = likelihood(output).probs.mean(0)[:, 1]
                pred_label = pred_prob.ge(0.5).float()
            else:
                if num_classes == 2:
                    test_loss += mll(output, test_y)
                    pred_prob = torch.sigmoid(output)
                    pred_label = pred_prob.ge(0.5).float()
                else:
                    test_loss += mll(output, test_y.long())
                    pred_label = torch.argmax(output, -1)

            test_auroc += roc_auc_score(test_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
            test_auprc += average_precision_score(test_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
            correct_predictions = pred_label.eq(test_y.view_as(pred_label)).cpu().numpy()
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
    log_dict = {'test_loss': test_loss, 'test_acc': test_acc,
                'test_auroc': test_auroc, 'test_auprc': test_auprc}

    if FLAGS.wandb_track:
        wandb.log(log_dict)
        fpr, tpr, _ = roc_curve(test_y.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
        data = [[x, y] for (x, y) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns=["fpr", "tpr"])
        wandb.log({
            "roc_curve": wandb.plot.line(
                table, "fpr", "tpr", title="Receiver operating characteristic")
        })

        # Save all checkpoints to wandb
        wandb.save(ckpt_path + '/*')

if __name__ == '__main__':
  app.run(main)
