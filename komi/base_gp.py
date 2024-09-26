from functools import reduce #, lru_cache
from typing import Union, List, Tuple
import warnings
import numpy as np
import torch
import gpytorch as gp
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

from .utilities import handle_covar_ 

class ExactGPModel(gp.models.ExactGP):
    """
    Standard exact GP model. Can handle independant multitasking via batch dimensions
    """
    def __init__( self,
                  train_x:Tensor,
                  train_y:Tensor,
                  likelihood:Union[Likelihood,None]=None,
                  n_tasks: int = 1,
                  kernel_type:Kernel=gp.kernels.RBFKernel,
                  mean_type:Mean=gp.means.ConstantMean,
                  decomp:Union[List[List[int]], None]=None,
                  outputscales:bool=False,
                  noise_thresh:float=1e-6,
                  n_inducing_points:Union[int,None]=None,
                  batch_lik:bool=False,
                  prior_scales:Union[Tensor, None]=None,
                  prior_width:Union[Tensor, None]=None,
                  ker_kwargs:Union[dict,None]=None,
                  jitter_val:float=1e-6,
                  **kwargs ):
        """
        Args:
            train_x: training input data
            train_y: training data labels
            likelihood: likelihood function for the model. If None, a Gaussian likelihood is used. Defaults to None.
            n_tasks: number of output tasks. Defaults to 1.
            kernel_type: . gp kernel function for latent processes. Defaults to gp.kernels.RBFKernel.
            mean_type: gp mean function for the outputs. Defaults to gp.means.ConstantMean.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            outputscales: whether to endow the kernel with a learned scaling factor, k(.) = a*k_base(.). Defaults to True
            noise_thresh: minimum value for the noise parameter. Has a large impact for ill-conditioned kernel matrices, which is the case of the HXS application. Defaults to 1e-6.
            n_inducing_points: if an integer is provided, the model will use the sparse GP approximation of Titsias (2009) with this many inducing points. Defaults to None.
            batch_lik: whether to use a batch Gaussian likelihood, or a MultitaskGaussianLikelihood able to model cross-tasks correlatins. Only meaningful if n_tasks > 1. Defaults to False.
            prior_scales: Prior mean for characteristic lengthscales of the kernel. Defaults to None.
            prior_width: Prior deviation-to-mean ratio for characteristic lengthscales of the kernel. Defaults to None.
            ker_kwargs: Additional arguments to pass to the gp kernel function. Defaults to None.
            jitter_val: jitter value for the Cholesky decomposition of the kernel matrix in specific leave-one-out computations. Defaults to 1e-6.
        """
        if likelihood is None:
            if n_tasks == 1:
                likelihood = gp.likelihoods.GaussianLikelihood(noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.noise = noise_thresh
            elif batch_lik :
                likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_tasks]),
                                                    noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.noise = noise_thresh * torch.ones_like(likelihood.noise)
            else:
                likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks,
                                                    noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.noise = noise_thresh
                likelihood.task_noises = torch.ones(n_tasks, device=train_y.device) * noise_thresh
                
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if ker_kwargs is None:
            ker_kwargs = {}
        self.dim = train_x.shape[1]
        self.n_tasks = n_tasks
        self.batch_lik = isinstance(likelihood, gp.likelihoods.GaussianLikelihood)
        self.mean_module = mean_type(input_size=self.dim, batch_shape=torch.Size([n_tasks]))
        self.covar_module = handle_covar_(kernel_type, dim=self.dim, decomp=decomp, prior_scales=prior_scales,
                                          prior_width=prior_width, outputscales=outputscales,
                                          n_funcs=n_tasks, ker_kwargs=ker_kwargs)
        if n_inducing_points is not None:
            self.covar_module = gp.kernels.InducingPointKernel(self.covar_module, torch.randn(n_inducing_points, self.dim), likelihood)
        
        if jitter_val is None:
            self.jitter_val = gp.settings.cholesky_jitter.value(train_x.dtype)
        else:
            self.jitter_val = jitter_val


    def forward( self, x:Tensor )-> Union[gp.distributions.MultivariateNormal, gp.distributions.MultitaskMultivariateNormal]:
        """
        Defines the computation performed at every call.
        Args:
            x: Data to evaluate the model at

        Returns:
            Prior distribution of the model output at the input locations. Can be a multitask multivariate normal if batch dimension is >1 
            and the model was instanciated with batch_lik = False, or a (batch) multivariate normal otherwise
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if not self.batch_lik and self.n_tasks > 1 : # for the batch case, but not the projected model inheritance
            return gp.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gp.distributions.MultivariateNormal(mean_x, covar_x))
        else:
            return gp.distributions.MultivariateNormal(mean_x, covar_x)


    def lscales( self, unpacked:bool=True )-> Union[List[Tensor], Tensor]:  # returned format : n_kernels x n_dims
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. 
            Applies only if the model kernel is not composite. Defaults to True

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_tasks x n_dims (n_dim number of dimensions of the subkernel)
        """
        if hasattr(self.covar_module, 'kernels'):
            n_kernels = len(self.covar_module.kernels)
            ref_kernel = self.covar_module.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = [reduce(getattr, attr_name.split('.'), ker).squeeze() for ker in self.covar_module.kernels]
        else:
            n_kernels = 1
            ref_kernel = self.covar_module
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = reduce(getattr, attr_name.split('.'), self.covar_module).squeeze()

        return [scales] if (n_kernels==1 and not unpacked) else scales

    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor. Defaults to False

        Returns:
            A tensor representing the learned outputscales of each subkernel and each task (shape n_tasks x n_kernels)
        """
        n_kernels = len(self.covar_module.kernels) if hasattr(self.covar_module, 'kernels') else 1
        n_funcs = self.n_latents if hasattr(self, 'n_latents') else self.n_tasks  ## to distinguish between the projected and batch-exact cases
        res = torch.zeros((n_funcs, n_kernels))
        if n_kernels > 1 :
            for i_ker in range(n_kernels):
                res[:, i_ker] = self.covar_module.kernels[i_ker].outputscale.data.squeeze()
        else:
            res[:,0] = self.covar_module.outputscale.data.squeeze()
        return res.squeeze() if (n_kernels==1 and unpacked) else res
    
    def kernel_cond( self ) -> Tensor:
        """
        Computes the condition number of the training kernel matrix.
        Returns:
            The condition number of the training kernel matrix
        """
        K_plus = self.prediction_strategy.lik_train_train_covar.evaluate_kernel().to_dense()
        return torch.linalg.cond(K_plus)
    
    def compute_loo(self, output=None, complex_mean=False) -> Tuple[Tensor, Tensor]:
        """
        Computes the leave-one-out (LOO) variance and error gaps (y_true - y_loo) values for the model.
        Args:
            output: model output at the training points, which may have been previously computed (for instance during training).
            If None, it is computed from the model. Defaults to None.
            complex_mean: whether to account for the variance of the mean function, in the case where it is a sum of regressors. Defaults to False.
        Returns:
            A tuple containing the LOO variances and error gaps for each task (each of size n_points x n_tasks)
        """
        train_x, train_y = self.train_inputs[0], self.train_targets
        likelihood = self.likelihood
        eps = self.jitter_val
        if self.n_tasks > 1:
            loo_var, loo_delta = torch.zeros_like(train_y), torch.zeros_like(train_y)
            ## The following blocks are leads for a more efficient implementation of the LOO computation in the multitask case. 
            ## Not yet functional because of bugs in gpytorch
            ###
            # K = likelihood(output).to_data_independent_dist().lazy_covariance_matrix
            # Kbatch = output.lazy_covariance_matrix.base_linear_op.detach().to_dense() # to be removed later. A bug in gpytorch makes this necessary
            ###
            # K = likelihood(output).lazy_covariance_matrix
            # identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            # L = K.cholesky(upper=False)
            # loo_var = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
            # loo_delta = L._cholesky_solve(targets.unsqueeze(-1), upper=False).squeeze(-1) * loo_var
            # loo_var, loo_delta = loo_var.detach().T, loo_delta.detach().T
            ###
            if hasattr(output.lazy_covariance_matrix, 'base_linear_op'):
                Kbatch = output.lazy_covariance_matrix.base_linear_op.evaluate() # to be removed later. A bug in gpytorch makes this necessary
            else:
                Kbatch = output.lazy_covariance_matrix.evaluate()
            global_noise = likelihood.noise.squeeze().data if hasattr(likelihood, 'noise') else 0
            m = self.mean_module(train_x).reshape(*train_y.shape)
            targets = (train_y - m).T
            for i in range(self.n_tasks):
                K = Kbatch[i]
                noise = global_noise + likelihood.task_noises.squeeze().data[i]
                identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
                K += noise * identity
                while eps < 1e6 * self.jitter_val: # this is a hack to avoid numerical instability
                    try:
                        L = K.cholesky(upper=False)
                        break
                    except:
                        eps *= 10
                        K += eps * identity
                        warnings.warn('Cholesky decomposition failed. Increasing jitter to {}'.format(eps))
                L = torch.linalg.cholesky(K, upper=False)
                loo_var[:,i] = 1.0 / torch.cholesky_solve(identity[None,:], L, upper=False).diagonal(dim1=-1, dim2=-2)
                loo_delta[:,i] = torch.cholesky_solve(targets[i].unsqueeze(-1), L, upper=False).squeeze(-1) * loo_var[:,i]
            loo_var, loo_delta = loo_var.detach(), loo_delta.detach()

        else: # single-output case
            # m, K, noise_it = self.mean_module(train_x), output.lazy_covariance_matrix, self.likelihood.noise.data
            m, K = self.mean_module(train_x), self.likelihood(output).lazy_covariance_matrix
            m = m.reshape(*train_y.shape)
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            with gp.settings.cholesky_max_tries(3):
                if complex_mean:
                    if not hasattr(self.mean_module, 'basis_matrix'):
                        raise ValueError('A complex mean treatment was required, but the model mean function doesn\'t allow it !')
                    else: # This has not been thoroughly tested yet
                        K_factors = K.cholesky(upper=False)
                        K_inv = K_factors._cholesky_solve(identity, upper=False).squeeze()
                        H = self.mean_module.basis_matrix(train_x)
                        M = torch.mm(torch.mm(H.T, K_inv), H)
                        M_factors = torch.linalg.cholesky(M + eps, upper=False)  # Now this is a Torch method, not a gp one (M is not a lazy matrix anymore)
                        identity_bis = torch.eye(*M.shape[-2:], dtype=M.dtype, device=M.device)
                        M_inv = torch.cholesky_solve(identity_bis, M_factors, upper=False)
                        K_minus = K_inv - K_inv @ H @ M_inv @ H.T @ K_inv
                        loo_var = 1.0 / K_minus.diagonal(dim1=-1, dim2=-2)
                        loo_delta = K_minus @ train_y * loo_var
                else:
                    L = K.cholesky(upper=False)
                    loo_var = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)
                    loo_delta = L._cholesky_solve((train_y - m).unsqueeze(-1), upper=False).squeeze(-1) * loo_var

        return loo_var, loo_delta
    
    def default_mll(self) -> MarginalLogLikelihood:
        """
        Returns the default marginal log likelihood (loss function) object for the model.
        Returns:
            A MarginalLogLikelihood object for the model
        """
        return gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)