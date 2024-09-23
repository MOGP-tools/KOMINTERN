from functools import reduce #, lru_cache
from typing import Union, List
import warnings
import torch
from torch import Tensor
import gpytorch as gp
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.operators.dense_linear_operator import to_linear_operator
from sklearn.decomposition import TruncatedSVD

from .base_gp import ExactGPModel
from .utilities import SplineKernel

class LazyLMCModel(ExactGPModel):
    """
    A training-less LMC-like model.
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, n_latents:int,
                  noise_val:float=1e-7,
                  store_full_y:bool=False,
                  ker_kwargs:Union[dict,None]=None,
                  jitter_val:Union[float,None]=None,
                  **kwargs):
        """
        Args:
            train_x: training input data
            train_y: training input labels
            n_latents: number of latent processes
        """
        n_points, n_tasks = train_y.shape
        proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_latents]),
                                        noise_constraint=gp.constraints.GreaterThan(noise_val))
        proj_likelihood.noise = noise_val
        
        SVD = TruncatedSVD(n_components=n_latents)
        y_transformed = SVD.fit_transform(train_y.cpu().T) #shape : n_tasks * n_latents
        proj_y = torch.as_tensor(SVD.components_.T) #shape : n_points * n_latents
        super().__init__(train_x=train_x, train_y=proj_y.T, likelihood=proj_likelihood, n_tasks=n_latents, kernel_type=SplineKernel,
                         mean_type=gp.means.ZeroMean, outputscales=False, n_inducing_points=None, **kwargs) # !! proj_likelihood will only be named likelihood in the model

        self.register_buffer('lmc_coeffs', torch.as_tensor(y_transformed.T, device=train_y.device))
        if store_full_y:
            self.register_buffer('train_y', train_y)
        self.full_lik = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks,
                                                    noise_constraint=gp.constraints.GreaterThan(noise_val))
        self.full_lik.noise = noise_val
        self.full_lik.task_noises = noise_val
        n_data, n_tasks = train_y.shape
        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.noise_val = noise_val
        self.latent_dim = -1
        if jitter_val is None:
            self.jitter_val = gp.settings.cholesky_jitter.value(train_x.dtype)
        else:
            self.jitter_val = jitter_val


    def projected_noise( self )-> Tensor:
        """
        Returns a vector of size n_latents containing the modeled noises of latent processes. Its diagonal embedding is the matrix Sigma_P from the article. 
        """
        return self.likelihood.noise.squeeze(-1)
    
    # @lru_cache(maxsize=None) # caching projected data and projected matrix is appealing, but it messes with backpropagation. No workaround has been found yet
    def projection_matrix( self )-> Tensor:
        """
        Returns matrix T from the article of shape n_tasks x n_latents, such that YT is the "projected data" seen by latent processes 
        """
        return self.lmc_coefficients.T

    def project_data( self, data):
        if hasattr(self, 'train_y') and data is self.train_y:
            return self.train_targets
        else:
            return (data @ self.projection_matrix()).T # shape n_latents x n_points ; opposite convention to most other quantities !!

    def full_likelihood( self ):
        return self.full_lik
    
    def lmc_coefficients( self ):
        return self.lmc_coeffs

    def forward( self, x:Tensor )-> gp.distributions.MultivariateNormal:  # ! forward only returns values of the latent processes !
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def compute_latent_distrib( self, x:Tensor, **kwargs )-> gp.distributions.MultivariateNormal:
        """
        Outputs (distributional) posterior values of the latent processes at the input locations. This is the function which is called to compute
        the loss during training.
        Args:
            x: input data tensor

        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        return ExactGPModel.__call__(self, x, **kwargs)  # shape n_latents x n_points
    
    def compute_loo(self, output=None, latent=False, train_y=None):
        if not latent:
            if hasattr(self, 'train_y'):
                train_y = self.train_y
            elif train_y is None:
                raise ValueError("The training labels must be provided to compute the task-level leave-one-out errors.\
                                 You can either provide them as an argument or store them in the model by instanciating it with store_full_y=True.")
        train_x = self.train_inputs[0]
        with torch.no_grad(), gp.settings.cholesky_max_tries(10):
            if output is None:
                output = self.compute_latent_distrib(train_x)
            K = self.likelihood(output).lazy_covariance_matrix
            y_proj = self.train_targets
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            L = K.cholesky(upper=False)
            loo_var = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
            loo_delta = L._cholesky_solve(y_proj.unsqueeze(-1), upper=False).squeeze(-1) * loo_var
            loo_var, loo_delta = loo_var.detach().T, loo_delta.detach().T
            if not latent:
                lmc_coeffs = self.lmc_coefficients()
                e_loo_raw = (loo_delta @ lmc_coeffs)
                diff = (train_y - y_proj.T @ lmc_coeffs)
                loo_delta = e_loo_raw + diff
                loo_var = loo_var @ lmc_coeffs**2
        return loo_var, loo_delta


    def save( self):
        dico = {}
        dico['lmc_coeffs'] = self.lmc_coeffs.tolist()
        dico['noise'] = self.noise_val
        with torch.no_grad():
            _ = self(torch.zeros_like(self.train_inputs[0])) # this is to compute the mean cache
        dico['mean_cache'] = self.prediction_strategy.mean_cache.tolist()
        return dico


    def __call__(self, x:Tensor, **kwargs)-> gp.distributions.MultitaskMultivariateNormal:
        """
        Outputs the full posterior distribution of the model at input locations. This is used to make predictions.
        Args:
            x: input data tensor

        Returns:
            A multitask multivariate gp normal distribution representing task processes values, which mean has shape n_points x n_tasks.
        """
        if self.training: # in training mode, we just compute the prior distribution of latent processes
            return super().__call__(x, **kwargs)
        
        latent_dist = ExactGPModel.__call__(self, x, **kwargs)

        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        num_dim = num_batch + len(latent_dist.event_shape)
        lmc_coefficients = self.lmc_coefficients().expand(*latent_dist.batch_shape, self.lmc_coefficients().size(-1))

        # Mean: ... x N x n_tasks
        latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
        mean = latent_mean @ lmc_coefficients.permute(
            *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
        )

        # Covar: ... x (N x n_tasks) x (N x n_tasks)
        latent_covar = latent_dist.lazy_covariance_matrix
        lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
        latent_covar = to_linear_operator(latent_covar.evaluate())
        covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
        covar = covar.add_jitter(self.jitter_val)

        return gp.distributions.MultitaskMultivariateNormal(mean, covar)

