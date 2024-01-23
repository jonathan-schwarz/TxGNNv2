"""Train and Fine-tune TxGNN model."""
from absl import app
from absl import flags

import math
import numpy as np
import os
import torch
import gpytorch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import GridInterpolationVariationalStrategy, UnwhitenedVariationalStrategy
from scipy.stats import entropy
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

DATA_PATH = '/n/home06/jschwarz/data/TxGNN/'

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
flags.DEFINE_integer('final_dim', 256, 'DKL Final Dim.', lower_bound=1)
flags.DEFINE_integer('hidden_dim', 256, 'DKL Hidden Dim.', lower_bound=1)
flags.DEFINE_integer('n_inducing', 100, 'Number of inducing points.', lower_bound=1)
flags.DEFINE_integer('n_layers', 3, 'DKL Hidden Layers.', lower_bound=1)


# Misc
flags.DEFINE_boolean('wandb_track', False, 'Use WandB')
# Valid choices are ['did', 'dod', 'dcd', 'drid', 'drod', 'drcd']
flags.DEFINE_string('dataset', 'txgnn_did', 'Dataset.')
flags.DEFINE_integer('seed', 42, 'Random seed.', lower_bound=0)
flags.DEFINE_string('checkpoint', './checkpoints/model_ckpt', 'Checkpoint location.')
flags.DEFINE_integer('valid_every', 25, 'Validation every #steps.', lower_bound=1)
flags.DEFINE_string('data_path', './data', 'Data location.')
flags.DEFINE_string('exp_name', 'debug', 'Experiment name.')

DD_TYPES = {
    'did': 'drug_indication_disease', 
    'dod': 'drug_off-label use_disease',
    'dcd': 'drug_contraindication_disease',
    'drid': 'disease_rev_indication_drug',
    'drod': 'disease_rev_off-label use_drug',
    'drcd': 'disease_rev_contraindication_drug',
}

LABELS_TO_DD_TYPES = {
    0: 'drug_indication_disease',
    1: 'drug_off-label use_disease',
    2: 'drug_contraindication_disease',
    3: 'disease_rev_indication_drug',
    4: 'disease_rev_off-label use_drug',
    5: 'disease_rev_contraindication_drug',
}


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, n_layers=3, hidden_dim=100, final_dim=2):
        super(FeatureExtractor, self).__init__()
        if n_layers == 1:
            self.add_module('linear1', torch.nn.Linear(data_dim, final_dim))
        else:
            self.add_module('linear1', torch.nn.Linear(data_dim, hidden_dim))
            self.add_module('relu1', torch.nn.ReLU())
            for i in range(1, n_layers):
                self.add_module('linear{}'.format(i+1), torch.nn.Linear(hidden_dim, hidden_dim))
                self.add_module('relu{}'.format(i+1), torch.nn.ReLU())
            self.add_module('linear4', torch.nn.Linear(hidden_dim, final_dim))


class GPClassificationModel(ApproximateGP):
    def __init__(self, dataset, num_dim, strategy,
                 grid_bounds, grid_size=64, inducing_x=None):
        if 'toy' == dataset:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_x.size(0))
            variational_strategy = UnwhitenedVariationalStrategy(
                self, inducing_points=inducing_x, 
                variational_distribution=variational_distribution,
                learn_inducing_locations=False
            )
            super(GPClassificationModel, self).__init__(variational_strategy)

            # if self.feature_extractor is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            #else:
            #    self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            #        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            #        num_dims=2, grid_size=100
            #    )
        elif False:#'txgnn_d' in dataset:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
            )
            variational_strategy = GridInterpolationVariationalStrategy(
                    self, grid_size=grid_size, grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
            )        
            super(GPClassificationModel, self).__init__(variational_strategy)

            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                        math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                    )
                )
            )
        else:
            # https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
            )
            if 'grid_interpolation' == strategy:
                variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    GridInterpolationVariationalStrategy(
                        self, grid_size=grid_size, grid_bounds=[grid_bounds],
                        variational_distribution=variational_distribution,
                    ), num_tasks=num_dim,
                )
            elif 'unwhitened' == strategy:
                # TODO(schwarzjn): This needs to be tested
                assert inducing_x is not None
                variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    UnwhitenedVariationalStrategy(
                        self, inducing_points=inducing_x, 
                        variational_distribution=variational_distribution,
                        learn_inducing_locations=False
                    ), num_tasks=num_dim,
                )                
            super(GPClassificationModel, self).__init__(variational_strategy)

            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                        math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                    )
                )
            )

        self.mean_module = gpytorch.means.ConstantMean()


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    

class DKLModel(gpytorch.Module):
    def __init__(self, dataset, feature_extractor, strategy, num_dim, grid_bounds=(-10., 10.),
                 inducing_x=None):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPClassificationModel(
            dataset=dataset, strategy=strategy,
            num_dim=num_dim, grid_bounds=grid_bounds,
            inducing_x=inducing_x)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim
        self.dataset = dataset

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
        

class DistMultModel(torch.nn.Sequential):
    def __init__(self, data_dim, num_classes=2):
        super(DistMultModel, self).__init__()
        self.data_dim = data_dim
        self.num_classes = num_classes
        if self.num_classes  > 2:
            self.W = torch.nn.Parameter(torch.Tensor(data_dim, num_classes))
            torch.nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain('linear'))
        else:
            self.W = torch.nn.Parameter(torch.Tensor(data_dim))
            torch.nn.init.zeros_(self.W)
        
    def forward(self, x):
        h_u, h_v = x.split(self.data_dim, dim=1)
        if self.num_classes  > 2:
            res = (h_u.unsqueeze(-1) * self.W * h_v.unsqueeze(-1)).sum(dim=1)
        else:
            res = (h_u * self.W * h_v).sum(dim=1).unsqueeze(-1)
        return res


def main(argv):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device( "cpu")


    if FLAGS.wandb_track:
        import wandb
        wandb.init(project='TxGNNv2', name='{}_finetune ({})'.format(FLAGS.dataset, FLAGS.model))

    if FLAGS.dataset == 'toy':
        num_train_points = 100
        num_valid_points = 50
        num_test_points = 1000

        train_x = torch.rand(num_train_points).unsqueeze(-1).to(device)
        train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).add(1).div(2).to(device)
        train_set = torch.utils.data.TensorDataset(train_x, train_y)

        valid_x = torch.rand(num_valid_points).unsqueeze(-1).to(device)
        valid_y = torch.sign(torch.cos(valid_x * (4 * math.pi))).add(1).div(2).to(device)
        valid_set = torch.utils.data.TensorDataset(valid_x, valid_y) 

        test_x = torch.linspace(0, 1, num_test_points).unsqueeze(-1).to(device)
        test_y = torch.sign(torch.cos(test_x * (4 * math.pi))).add(1).div(2).to(device)
        test_set = torch.utils.data.TensorDataset(test_x, test_y)

        data_dim = 1
        num_classes = 2
        inducing_x = train_x
    elif FLAGS.dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        common_trans = [transforms.ToTensor()]#, normalize]
        train_compose = transforms.Compose(common_trans)
        test_compose = transforms.Compose(common_trans)

        train_set = dset.MNIST('data', train=True, transform=train_compose, download=True)
        valid_set = dset.MNIST('data', train=False, transform=test_compose)
        test_set = dset.MNIST('data', train=False, transform=test_compose)

        data_dim = 28*28
        num_classes = 10
        num_valid_points = 10000
        num_test_points = 10000
        inducing_x = None
    elif FLAGS.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        common_trans = [transforms.ToTensor(), normalize]
        train_compose = transforms.Compose(aug_trans + common_trans)
        test_compose = transforms.Compose(common_trans)

        train_set = dset.CIFAR10('data', train=True, transform=train_compose, download=True)
        valid_set = dset.CIFAR10('data', train=False, transform=test_compose)
        test_set = dset.CIFAR10('data', train=False, transform=test_compose)

        data_dim = 32*32*3
        num_classes = 10
        num_valid_points = 10000
        num_test_points = 10000
        inducing_x = None
    elif FLAGS.dataset == 'txgnn_all':
        data = np.load(os.path.join(DATA_PATH, 'pretrained/together.npz'))

        train_x = torch.Tensor(np.concatenate(
            [data['train_h_u'], data['train_h_v']], axis=1)).to(device)
        inducing_x = train_x
        train_y = torch.Tensor(data['train_labels']).to(device)   
        train_set = torch.utils.data.TensorDataset(train_x, train_y)

        valid_x = torch.Tensor(np.concatenate(
            [data['valid_h_u'], data['valid_h_v']], axis=1)).to(device)
        valid_y = torch.Tensor(data['valid_labels']).to(device) 
        valid_set = torch.utils.data.TensorDataset(valid_x, valid_y)

        test_x = torch.Tensor(np.concatenate(
            [data['test_h_u'], data['test_h_v']], axis=1)).to(device)
        test_y = torch.Tensor(data['test_labels']).to(device)    
        test_set = torch.utils.data.TensorDataset(test_x, test_y)

        data_dim = train_x.shape[1]
        num_classes = 12
        num_valid_points = valid_x.shape[0]
        num_test_points = test_x.shape[0]
        inducing_x = None
    elif 'txgnn_d' in FLAGS.dataset:
        path = os.path.join(
            DATA_PATH, 'pretrained_mine/complex_disease/separate/{}.npz')

        data = np.load(path.format(
            DD_TYPES[FLAGS.dataset.split('_')[1]]))

        train_x = torch.Tensor(np.concatenate(
            [data['train_h_u'], data['train_h_v']], axis=1)).to(device)
        train_y = torch.Tensor(data['train_labels']).to(device)   
        train_names = np.concatenate(
            [data['train_u_names'], data['train_v_names']], axis=1)
        train_set = torch.utils.data.TensorDataset(train_x, train_y) 

        valid_x = torch.Tensor(np.concatenate(
            [data['valid_h_u'], data['valid_h_v']], axis=1)).to(device)
        valid_y = torch.Tensor(data['valid_labels']).to(device) 
        valid_names = np.concatenate(
            [data['valid_u_names'], data['valid_v_names']], axis=1)
        valid_set = torch.utils.data.TensorDataset(valid_x, valid_y) 

        test_x = torch.Tensor(np.concatenate(
            [data['test_h_u'], data['test_h_v']], axis=1)).to(device)
        test_y = torch.Tensor(data['test_labels']).to(device)     
        test_names = np.concatenate(
            [data['test_u_names'], data['test_v_names']], axis=1)
        test_set = torch.utils.data.TensorDataset(test_x, test_y) 

        data_dim = train_x.shape[1]
        num_classes = 2
        num_valid_points = valid_x.shape[0]
        num_test_points = test_x.shape[0]
        inducing_x = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=FLAGS.batch_size, shuffle=True)
    num_train_points = len(train_loader.dataset)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=num_valid_points, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=num_test_points, shuffle=False)

    likelihood = None
    if 'dkl' == FLAGS.model:
        # Initialize model and likelihood
        feature_extractor = FeatureExtractor(
            data_dim=data_dim, 
            hidden_dim=FLAGS.hidden_dim, 
            n_layers=FLAGS.n_layers, 
            final_dim=FLAGS.final_dim).to(device)
        model = DKLModel(FLAGS.dataset, feature_extractor, strategy=FLAGS.strategy,
                         num_dim=FLAGS.final_dim, inducing_x=inducing_x).to(device)
        # TODO(schwarzjn): Use BernoulliLikelihood
        #if num_classes == 2:
        #    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
        #else:
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
                num_features=model.num_dim, num_classes=num_classes).to(device)
    elif 'distmult' == FLAGS.model:
        model = DistMultModel(data_dim//2, num_classes).to(device)
    elif 'mlp' == FLAGS.model:
        model = FeatureExtractor(
            data_dim=data_dim, 
            hidden_dim=FLAGS.hidden_dim, 
            n_layers=FLAGS.n_layers, 
            final_dim=num_classes if num_classes > 2 else 1).to(device)

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=5)

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
                    pred_prob = likelihood(output).probs.mean(0)
                    pred_label = pred_prob.argmax(-1)

                    train_y_ = train_y.detach().cpu().numpy()
                    pred_prob_ = pred_prob.detach().cpu().numpy()
                    rel_pred_prob_ = 1.0 - pred_prob_[
                        range(train_y_.shape[0]), train_y_.astype(np.int32).flatten()]

                    train_auroc = roc_auc_score(train_y_, rel_pred_prob_)
                    train_auprc = average_precision_score(train_y_, rel_pred_prob_)
                else:
                    if num_classes == 2:
                        loss = mll(output, train_y)
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
                    # all_prediction = []
                    # all_labels = []         
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
                                pred_prob = likelihood(output).probs.mean(0)
                                pred_label = pred_prob.argmax(-1)

                                valid_y_ = valid_y.detach().cpu().numpy()
                                pred_prob_ = pred_prob.detach().cpu().numpy()
                                rel_pred_prob_ = 1.0 - pred_prob_[
                                    range(valid_y_.shape[0]), valid_y_.astype(np.int32).flatten()]
                                valid_auroc += roc_auc_score(valid_y_, rel_pred_prob_)
                                valid_auprc += average_precision_score(valid_y_, rel_pred_prob_)
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
                            # all_logits.append(output.detach().cpu().numpy())
                            # all_labels.append(valid_y.detach().cpu().numpy())
                            n_batches += 1

                    # Track Validation loss
                    valid_acc = 100. * valid_correct.item() / float(len(valid_loader.dataset))
                    valid_loss = (valid_loss  / float(n_batches)).item()
                    valid_auroc = (valid_auroc  / float(n_batches))
                    valid_auprc = (valid_auprc  / float(n_batches))

                    print('Epoch/Iter: {}/{} - Valid Loss: {:.3f} - Acc: {}/{} ({:.3f}%) - AUROC: {:.3f} - AUPRC: {:.3f}'.format(
                        i + 1, j + 1, valid_loss, valid_correct.item(), len(valid_loader.dataset), valid_acc, valid_auroc, valid_auprc)
                    )

                    wandb_log_dict = {'valid_loss': valid_loss, 'valid_acc': valid_acc, 
                                      'valid_auroc': valid_auroc, 'valid_auprc': valid_auprc}
                    if FLAGS.wandb_track:
                        wandb.log(wandb_log_dict)

                    scheduler.step(valid_loss)

                j += 1

    model.eval()
    if likelihood is not None:
        likelihood.eval()

    test_auprc = 0.0
    test_auroc = 0.0
    test_correct = 0
    test_entropy = 0.0
    test_loss = 0.0
    # all_logits = []
    # all_labels = []
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
                pred_prob = likelihood(output).probs.mean(0)
                pred_label = pred_prob.argmax(-1)

                test_y_ = test_y.detach().cpu().numpy()
                pred_prob_ = pred_prob.detach().cpu().numpy()
                rel_pred_prob_ = 1.0 - pred_prob_[
                    range(test_y_.shape[0]), test_y_.astype(np.int32).flatten()][:, np.newaxis]
            else:
                if num_classes == 2:
                    test_loss += mll(output, test_y)
                    pred_prob = torch.sigmoid(output)
                    pred_label = pred_prob.ge(0.5).float()
                else:
                    test_loss += mll(output, test_y.long())                    
                    pred_label = torch.argmax(output, -1)

                test_y_ = test_y.detach().cpu().numpy()
                rel_pred_prob_ = pred_prob.detach().cpu().numpy()

            #all_logits.append(output.detach().cpu().numpy())
            #all_labels.append(test_y.detach().cpu().numpy())

            test_auroc += roc_auc_score(test_y_, rel_pred_prob_)
            test_auprc += average_precision_score(test_y_, rel_pred_prob_)
            correct_predictions = pred_label.eq(test_y.view_as(pred_label)).cpu().numpy()
            test_correct += correct_predictions.sum()
            test_entropy += entropy(
                np.concatenate([rel_pred_prob_, 1 - rel_pred_prob_], axis=1), axis=1).sum()
            n_batches += 1

    # Track Test loss
    test_acc = 100. * test_correct / float(len(test_loader.dataset))  
    test_loss = (test_loss  / float(n_batches)).item()
    test_auroc = (test_auroc  / float(n_batches))
    test_auprc = (test_auprc  / float(n_batches))
    test_entropy = (test_entropy  / float(len(test_loader.dataset)))
    
    print('{}. test loss: {:.3f} - acc: {}/{} ({:.3f}%) - auroc: {:.3f} - auprc: {:.3f} - entropy: {:.3f}'.format(
        FLAGS.dataset, test_loss, test_correct.item(), len(test_loader.dataset), test_acc, test_auroc, test_auprc, test_entropy)
    )
    wandb_log_dict = {'test_loss': test_loss, 'test_acc': test_acc, 
                      'test_auroc': test_auroc, 'test_auprc': test_auprc,
                      'test_entropy': test_entropy}

    if FLAGS.wandb_track:
        wandb.log(wandb_log_dict)

        fpr, tpr, _ = roc_curve(test_y_, rel_pred_prob_)
        data = [[x, y] for (x, y) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns=["fpr", "tpr"])
        wandb.log({
            "roc_curve": wandb.plot.line(
                table, "fpr", "tpr", title="Receiver operating characteristic")
        })

if __name__ == '__main__':
  app.run(main)
