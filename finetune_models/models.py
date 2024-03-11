import gpytorch
import math
import torch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import GridInterpolationVariationalStrategy, UnwhitenedVariationalStrategy 


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

    def forward(self, input):
        h_u, h_v = input.split(self.data_dim, dim=1)
        if self.num_classes  > 2:
            res = (h_u.unsqueeze(-1) * self.W * h_v.unsqueeze(-1)).sum(dim=1)
        else:
            res = (h_u * self.W * h_v).sum(dim=1).unsqueeze(-1)
        return res


class GPClassificationModel(ApproximateGP):
    def __init__(self, num_dim, strategy,
                 grid_bounds, grid_size=64, inducing_x=None):
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


    def forward(self, input):
        mean_x = self.mean_module(input)
        covar_x = self.covar_module(input)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class DKLModel(gpytorch.Module):
    def __init__(self, strategy, num_dim, grid_bounds=(-10., 10.), grid_size=64,
                 inducing_x=None, feature_extractor=None):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPClassificationModel(
            strategy=strategy, num_dim=num_dim, grid_size=grid_size,
            grid_bounds=grid_bounds, inducing_x=inducing_x)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, input):
        if self.feature_extractor is not None:
            input = self.feature_extractor(input)

        input = self.scale_to_bounds(input)
        # This next line makes it so that we learn a GP for each feature
        input = input.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(input)
        return res


def get_model(model_type, data_dim, num_classes, hidden_dim, n_layers, final_dim,
              strategy, inducing_x, device, use_feature_extractor=True):
    likelihood = None
    if 'dkl' in model_type:
        # Initialize model and likelihood
        if use_feature_extractor:
            feature_extractor = FeatureExtractor(
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                final_dim=final_dim).to(device)
        else:
            feature_extractor = None
            final_dim = data_dim

        model = DKLModel(feature_extractor=feature_extractor, strategy=strategy,
                         num_dim=final_dim, inducing_x=inducing_x).to(device)
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
                num_features=model.num_dim, num_classes=num_classes).to(device)
    elif 'distmult' == model_type:
        model = DistMultModel(data_dim//2, num_classes).to(device)
    elif 'mlp' == model_type:
        model = FeatureExtractor(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            final_dim=num_classes if num_classes > 2 else 1).to(device)

    return model, likelihood
