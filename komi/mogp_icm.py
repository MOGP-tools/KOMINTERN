from functools import reduce
from typing import Union, List
import psutil
import warnings
import copy
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.likelihoods.likelihood import Likelihood
from linear_operator.operators import PsdSumLinearOperator

from .utilities import init_lmc_coefficients
from .base_gp import ExactGPModel

class MultitaskGPModel(ExactGPModel):
    """
    A multitask GP model with exact GP treatment. This class encompasses both ICM and naive LMC models.
    """
    def __init__( self,
                  train_x: Tensor,
                  train_y: Tensor,
                  n_latents: int, 
                  likelihood:Union[Likelihood,None]=None,
                  fix_diagonal:bool=True,
                  diag_value:float=16*torch.finfo(torch.get_default_dtype()).tiny, 
                  noise_thresh:float=1e-4,
                  init_lmc_coeffs:bool=True,
                  outputscales:bool=False,
                  model_type:str='ICM',
                  **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel (in particular the choice of 
        mean and kernel function) also apply here thanks to the inheritance.

        Args:
            train_x: Input data
            train_y: Input labels
            n_latents: number of latent functions
            likelihood: gpytorch likelihood function for the outputs. If none is provided, a default MultitaskGaussianLikelihood is used. Defaults to None.
            fix_diagonal: for ICM only. If True, fixes the learned diagonal term of the task covariance matrix, accounting for task-specific (non-shared)
            latent processes. The efficient storage of the model (with a cache of size n_latents x n_points) is only possible if this diagonal term
            is fixed to zero. Defaults to True.
            diag_value: value of the diagonal term of the task covariance matrix if fix_diagonal is set to True. Defaults to machine precision.
            init_lmc_coeffs: whether to initialize LMC coefficients with SVD of the training labels. If False, these coefficients are sampled from a normal distribution. Defaults to True.
            outputscales: whether to endow each latent kernel with a learned scaling factor, k(.) = a*k_base(.). This is only useful for predictive variance 
            scaling, and may result in over-parametrization. Defaults to False
            model_type: choice between 'ICM' and 'LMC'. The latter is very computationnally-heavy and unstable, so it should only be used for very specific
            experimental purposes. Defaults to "ICM"
        """
        n_data, n_tasks = train_y.shape
        if likelihood is None:
            likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks,
                                                    noise_constraint=gp.constraints.GreaterThan(noise_thresh))
            likelihood.noise = noise_thresh
            likelihood.task_noises = torch.ones(n_tasks, device=train_y.device) * noise_thresh
            
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood, n_tasks=1, outputscales=outputscales, **kwargs) # we build upon a single-task GP, created by calling parent class

        self.mean_module = gp.means.MultitaskMean(self.mean_module, num_tasks=n_tasks)
        
        if model_type=='ICM':
            self.covar_module = gp.kernels.MultitaskKernel(self.covar_module, num_tasks=n_tasks, rank=n_latents)
        elif model_type=='LMC':
            self.covar_module = gp.kernels.LCMKernel(base_kernels=[copy.deepcopy(self.covar_module) for i in range(n_latents)],
                                                           num_tasks=n_tasks, rank=1)

        if init_lmc_coeffs:
            lmc_coeffs = init_lmc_coefficients(train_y, n_latents).T
            if model_type=='ICM':
                # this parameter has already been initialized with random values at the instantiation of the variational strategy, so registering it anew is facultative
                self.covar_module.task_covar_module.register_parameter(name='covar_factor', parameter=torch.nn.Parameter(lmc_coeffs))
            elif model_type=='LMC':
                for i in range(n_latents):
                    # this parameter has already been initialized with random values at the instantiation of the variational strategy, so registering it anew is facultative
                    self.covar_module.covar_module_list[i].task_covar_module.covar_factor = torch.nn.Parameter(lmc_coeffs[:,i].unsqueeze(-1))
            else:
                raise ValueError('Wrong specified model type, should be ICM or LMC')

        if fix_diagonal:
            if model_type=='ICM':
                self.covar_module.task_covar_module.register_parameter(name='raw_var',
                                            parameter=torch.nn.Parameter(np.log(diag_value)*torch.ones(n_tasks, device=train_y.device),
                                            requires_grad=False))
            elif model_type=='LMC':
                for i in range(len(self.covar_module.covar_module_list)):
                    self.covar_module.covar_module_list[i].task_covar_module.register_parameter(name='raw_var',
                                                parameter=torch.nn.Parameter(np.log(diag_value)*torch.ones(n_tasks, device=train_y.device),
                                                requires_grad=False))
                
        self.outputscales = outputscales
        self.n_tasks, self.n_latents, self.model_type = n_tasks, n_latents, model_type

    def lmc_coefficients( self )-> Tensor:
        """

        Returns:
            tensor of shape n_latents x n_tasks representing the LMC/ICM coefficients of the model.
        """
        if self.model_type=='LMC':
            res = torch.zeros((self.n_latents, self.n_tasks))
            for i in range(self.n_latents):
                res[i] = self.covar_module.covar_module_list[i].task_covar_module.covar_factor.data.squeeze()
        else:
            res = self.covar_module.task_covar_module.covar_factor.data.squeeze().T
        return res

    def lscales( self, unpacked:bool=True)-> Union[List[Tensor], Tensor] :
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. 
            Applies only if the model kernel is not composite. Defaults to True

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_latents x n_dims (n_dim number of dimensions of the subkernel)
        """
        if self.model_type=='LMC':
            data_covar = self.covar_module.covar_module_list[0].data_covar_module
        else:
            data_covar = self.covar_module.data_covar_module

        if hasattr(data_covar, 'kernels'):
            n_kernels = len(data_covar.kernels)
            ref_kernel = data_covar.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            if self.model_type=='ICM':
                scales = [reduce(getattr, attr_name.split('.'), ker).squeeze().repeat(self.n_latents, 1) for ker in data_covar.kernels]
            else:
                ref_scales = [reduce(getattr, attr_name.split('.'), ker).squeeze() for ker in data_covar.kernels]
                ker_dims = [len(scales) if scales.ndim > 0 else 1 for scales in ref_scales]
                scales = [torch.zeros((self.n_latents, ker_dims[i])) for i in range(n_kernels)]
        else:
            n_kernels = 1
            ref_kernel = data_covar
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            if self.model_type=='ICM':
                scales = reduce(getattr, attr_name.split('.'), data_covar).squeeze().repeat(self.n_latents, 1)
            else:
                ref_scales = reduce(getattr, attr_name.split('.'), data_covar).squeeze()
                ker_dim = len(ref_scales) if ref_scales.ndim > 0 else 1
                scales = torch.zeros((self.n_latents, ker_dim))

        if self.model_type=='LMC':
            for i in range(self.n_latents):
                if n_kernels > 1:
                    for j in range(n_kernels):
                        scales[j][i,:] = reduce(getattr, attr_name.split('.'), self.covar_module.covar_module_list[i].data_covar_module.kernels[j]).squeeze()
                else:
                    scales[i,:] = reduce(getattr, attr_name.split('.'), self.covar_module.covar_module_list[i].data_covar_module).squeeze()
      
        return [scales] if (n_kernels==1 and not unpacked) else scales    
        
    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor. Defaults to False

        Returns:
            A tensor representing the learned outputscales of each subkernel and each task (shape n_latents x n_kernels)
        """
        if self.model_type=='LMC':
            data_covar = self.covar_module.covar_module_list[0].data_covar_module
        else:
            data_covar = self.covar_module.data_covar_module

        n_kernels = len(data_covar.kernels) if hasattr(data_covar, 'kernels') else 1
        res = torch.zeros((self.n_latents, n_kernels))
        if n_kernels > 1:
            for i_ker in range(n_kernels):
                if self.model_type=='LMC':
                    for i_lat in range(self.n_latents):
                        res[i_lat, i_ker] = self.covar_module.covar_module_list[i_lat].data_covar_module.kernels[i_ker].outputscale.data.squeeze()
                else:
                    res[:, i_ker] = data_covar.kernels[i_ker].outputscale.data.squeeze()
        else:
            if self.model_type == 'LMC':
                for i_lat in range(self.n_latents):
                    res[i_lat, 0] = self.covar_module.covar_module_list[i_lat].data_covar_module.outputscale.data.squeeze()
            else:
                res[:, 0] = data_covar.outputscale.data.squeeze()

        return res.squeeze() if (n_kernels==1 and unpacked) else res

    def forward( self, x ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def compute_var(self, x):
        """
        Computes the variance of the model at input locations.
        Args:
            x: input data to evaluate the model at

        Returns:
            A tensor of the variances of the model at input locations.
        """
        if self.model_type!='ICM':
            raise ValueError('This method is only available for ICM models')
        
        linop = self.covar_module.forward(x,x) + self.likelihood._shaped_noise_covar((len(x), len(x)), add_noise=True)
        first_term = linop.diagonal(dim1=-2, dim2=-1).reshape((len(x), self.n_tasks))

        x_train = self.train_inputs[0]
        ker_op = self.covar_module.forward(x_train,x_train)
        noise_op = self.likelihood._shaped_noise_covar((len(x_train), len(x_train)), add_noise=True)
        i_task = 1 if isinstance(ker_op.linear_ops[1], PsdSumLinearOperator) else 0

        data_ker = ker_op.linear_ops[(i_task + 1) % 2]
        k_evals, k_evecs = data_ker._symeig(eigenvectors=True)

        noise_inv_root = noise_op.linear_ops[i_task].root_inv_decomposition()
        C = ker_op.linear_ops[i_task]
        C_tilde = noise_inv_root.matmul(C.matmul(noise_inv_root))
        C_evals, C_evecs = C_tilde._symeig(eigenvectors=True)
        C_hat = C.matmul(noise_inv_root).matmul(C_evecs).evaluate().squeeze()
        C_square = C_hat**2

        S = torch.kron(k_evals, C_evals) + 1.0
        if x.is_cuda:
            free_mem = torch.cuda.mem_get_info()[0]
        else:
            free_mem = psutil.virtual_memory()[1]

        num_bytes = x.element_size()
        batch_size = int(free_mem / (16 * len(x_train) * self.n_tasks**2 * num_bytes))
        n = x.shape[0]  # Total number of samples
        second_term_results = []  # List to store the results
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size]
            k_hat = self.covar_module.data_covar_module(x_batch, x_train).matmul(k_evecs).evaluate().squeeze()
            k_square = k_hat**2
            second_term = torch.kron(k_square, C_square) @ S.pow(-1).squeeze()
            second_term_results.append(second_term.reshape((len(x_batch), self.n_tasks)))

        # Convert the list of results to a tensor
        second_term = torch.cat(second_term_results)
        return torch.clamp(first_term - second_term, min=1e-6)

    #This function is very slow, it doesn't seem to fully leverage gpytorch's internals for efficient ICM algebra. To be improved
    def compute_loo(self, output=None):
        train_x, train_y = self.train_inputs[0], self.train_targets        
        with gp.settings.cholesky_max_tries(6), \
            torch.no_grad():
            if output is None:
                output = self.forward(train_x)
            m, K = self.mean_module(train_x), self.likelihood(output).lazy_covariance_matrix
            m = m.reshape(*train_y.shape)
            targets = torch.reshape(train_y - m, (np.prod(train_y.shape).astype(int),1))
            L = K.cholesky(upper=False)
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            loo_var = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)
            loo_delta = L._cholesky_solve(targets, upper=False).squeeze(-1) * loo_var
            loo_var = torch.reshape(loo_var.detach(), train_y.shape)
            loo_delta = torch.reshape(loo_delta.detach(), train_y.shape)
        return loo_var, loo_delta
    
    def save( self, extra_terms=False):
        self.eval()
        dico = {}
        dico['lmc_coeffs'] = self.lmc_coefficients().detach().tolist()
        if self.outputscales:
            dico['outputscales'] = self.outputscale().tolist()
        dico['lscales'] = self.lscales().tolist()
        with gp.beta_features.checkpoint_kernel(0),\
                gp.settings.skip_posterior_variances(state=True),\
                torch.no_grad():
            _ = self(torch.zeros_like(self.train_inputs[0])) # to compute the mean_cache
            gp_cache = self.prediction_strategy.mean_cache
            n_points, n_tasks = self.train_targets.shape
            res = gp_cache.reshape((n_points, n_tasks)).matmul(self.covar_module.task_covar_module.covar_matrix)
            dico['mean_cache'] = res.detach().tolist()

        if extra_terms:
            likelihood = self.likelihood
            noises = likelihood.noise.detach() if hasattr(likelihood, 'noise') else 0.
            if hasattr(likelihood, 'task_noises'):
                noises = self.likelihood.task_noises.detach() + noises
            elif hasattr(likelihood, 'task_noise_covar'):
                dico['task_noise_covar'] = likelihood.task_noise_covar.detach().tolist()
            dico['noises'] = noises.tolist()
        
        return dico

    def default_mll(self):
        return gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

