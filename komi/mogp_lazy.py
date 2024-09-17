from functools import reduce #, lru_cache
from typing import Union, List
import warnings
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.means.mean import Mean
from gpytorch.likelihoods.likelihood import Likelihood
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.operators.dense_linear_operator import to_linear_operator
from komi.utilities import init_lmc_coefficients, ScalarParam, PositiveDiagonalParam, LowerTriangularParam
from base_gp import ExactGPModel

class LazyLMCModel(ExactGPModel):
    """
    A training-less LMC-like model.
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, n_latents:int, mean_type:Mean=gp.means.ConstantMean, outputscales:bool=False, 
                  decomp:Union[List[List[int]],None]=None,
                  noise_val:float=1e-4,
                  ker_kwargs:Union[dict,None]=None, **kwargs):
        """
        Args:
            train_x: training input data
            train_y: training input labels
            n_latents: number of latent processes
            mean_type: gp mean function for task-level processes. Defaults to gp.means.ConstantMean.
        """
        proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_latents]),
                                        noise_constraint=gp.constraints.GreaterThan(np.exp(noise_val))) #useless, only for inheritance
        super().__init__(train_x, torch.zeros_like(train_y), proj_likelihood, n_tasks=n_latents, 
                         mean_type=gp.means.ZeroMean, outputscales=outputscales, **kwargs) # !! proj_likelihood will only be named likelihood in the model
        self.register_buffer('train_y', train_y)

        n_data, n_tasks = train_y.shape

        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.latent_dim = -1
