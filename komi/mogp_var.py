from functools import reduce #, lru_cache
from typing import Union, List, Tuple
import warnings
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from scipy.stats import qmc

from .utilities import init_lmc_coefficients, handle_covar_, compute_truncated_svd

class CustomLMCVariationalStrategy(gp.variational.LMCVariationalStrategy):
    """
    This small overlay to the native LMCVariationalStrategy of gp allows to put deterministic mean functions on tasks rather than latent processes.
    """
    def __init__(self, mean_module:gp.means.mean, *args, **kwargs):
        """

        Args:
            mean_module: The already-generated, batched, many-tasks mean function to impose on output tasks.
        """
        super().__init__(*args, **kwargs)
        self.output_mean_module = mean_module

    def __call__(self, x:Tensor, task_indices=None, prior=False, **kwargs)-> gp.distributions.MultitaskMultivariateNormal:
        """

        Args:
            x:Input data to evaluate model at.

        Returns:
            The posterior distribution of the model at input locations.
        """
        multitask_dist = super().__call__(x, task_indices=None, prior=False, **kwargs)
        tasks_means = self.output_mean_module(x)
        return multitask_dist.__class__(multitask_dist.mean + tasks_means.mT, multitask_dist.lazy_covariance_matrix)


class VariationalMultitaskGPModel(gp.models.ApproximateGP):
    """
    A standard variational LMC model using gp functionalities.
    """
    def __init__( self,
                 train_x:Tensor,
                 train_y:Tensor,
                 n_latents:int,
                 train_ind_ratio:float=1.5,
                 likelihood:Union[Likelihood,None]=None, 
                 kernel_type:Kernel=gp.kernels.RBFKernel,
                 mean_type:Mean=gp.means.ConstantMean,
                 decomp:Union[List[List[int]],None]=None,
                 disc_ranks:Tuple[int,...]=(),
                 distrib:gp.variational._VariationalDistribution=gp.variational.CholeskyVariationalDistribution, 
                 var_strat:gp.variational._VariationalStrategy=gp.variational.VariationalStrategy,
                 init_lmc_coeffs:bool=True,
                 noise_thresh:float=1e-4,
                 outputscales:bool=False, 
                 batch_lik:bool=False,
                 prior_scales:Tensor=None,
                 prior_width:Tensor=None,
                 ker_kwargs:Union[dict,None]=None, 
                 seed:int=0,
                 **kwargs):
        """
        Args:
            train_x: training input data
            train_y: training data labels, used only for the SVD initialization of LMC coefficients (with this model, data labels are only used 
            | during loss computation, not predictions)
            n_latents: number of latent processes
            train_ind_ratio: ratio between the number of training points and this of inducing points. Defaults to 1.5.
            likelihood: gpytorch likelihood function for the outputs. If none is provided, a default MultitaskGaussianLikelihood is used. Defaults to None.
            kernel_type: gpytorch kernel function for the latent processes. Defaults to gp.kernels.RBFKernel.
            mean_type: gpytorch mean function for the outputs. Defaults to gp.means.ConstantMean.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            disc_ranks: optional tuple of integers indicating the correlation rank of each discrete kernel.
            | Used (and required) only if decomp contains a non-empty 'cont' entry (continuous variables)
            distrib: gpytorch variational distribution for inducing values (see gpytorch documentation). Defaults to gp.variational.CholeskyVariationalDistribution.
            var_strat: gpytorch variational strategy (see gpytorch documentation). Defaults to gp.variational.VariationalStrategy.
            init_lmc_coeffs: whether to initialize LMC coefficients with SVD of the training labels. If False, these coefficients are sampled from a normal distribution. Defaults to True.
            noise_thresh: minimum value for the noise parameter. Has a large impact for ill-conditioned kernel matrices, which is the case of the HXS application. Defaults to 1e-6.
            outputscales: whether to endow each latent kernel with a learned scaling factor, k(.) = a*k_base(.). This is only useful for predictive variance 
            | scaling, and may result in over-parametrization. Defaults to False
            batch_lik: if True, the likelihood will be a batch of gaussians instead of a multitask independent gaussian. May improve perf
            prior_scales: Prior mean for characteristic lengthscales of the kernel. Defaults to None.
            prior_width: Prior deviation-to-mean ratio for characteristic lengthscales of the kernel. Defaults to None.
            ker_kwargs: Additional arguments to pass to the gp kernel function. Defaults to None.
            seed: Random seed for inducing points generation. Defaults to 0.
        """

        if ker_kwargs is None:
            ker_kwargs = {}

        *batch_shape, n_points, n_tasks = train_y.shape
        latent_batch_shape = torch.Size([*batch_shape, n_latents])
        output_batch_shape = torch.Size([*batch_shape, n_tasks])
        multilik_batch_shape = torch.Size(batch_shape)
        _, dim = train_x.shape

        if float(train_ind_ratio) == 1.:
            warnings.warn('Caution : inducing points not learned !')
            inducing_points = train_x
            learn_inducing_locations = False
            var_strat = gp.variational.UnwhitenedVariationalStrategy  #better compatibility in this case
            distrib = gp.variational.CholeskyVariationalDistribution  #better compatibility in this case
        else:
            learn_inducing_locations = True
            n_ind_points = int(np.floor(n_points / train_ind_ratio))
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
            inducing_points = torch.as_tensor(2 * sampler.random(n=n_ind_points) - 1, dtype=train_x.dtype)
            #same inducing points for all latents here

        variational_distribution = distrib(inducing_points.size(-2), batch_shape=latent_batch_shape)
        strategy = var_strat(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations)
        output_mean_module = mean_type(input_size=dim, batch_shape=output_batch_shape)

        variational_strategy = CustomLMCVariationalStrategy(
            output_mean_module,
            strategy,
            num_tasks=n_tasks,
            num_latents=n_latents,
            latent_dim=-1,
            **kwargs)

        super().__init__(variational_strategy)

        self.covar_module = handle_covar_(kernel_type, dim=dim, decomp=decomp, disc_ranks=disc_ranks, prior_scales=prior_scales,
                            prior_width=prior_width, batch_shape=latent_batch_shape, ker_kwargs=ker_kwargs, outputscales=outputscales)
        self.mean_module = gp.means.ZeroMean(batch_shape=latent_batch_shape) #in gp, latent processes can have non-zero means, which we wish to avoid

        if likelihood is None:
            noise_init = 10 * noise_thresh
            if batch_lik :
                likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=output_batch_shape, noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.noise = noise_init * torch.ones_like(likelihood.noise)
            else:
                likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, batch_shape=multilik_batch_shape,
                                                                        has_global_noise=False,
                                                                        noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.task_noises = torch.ones_like(likelihood.task_noises) * noise_init

        self.likelihood = likelihood
        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.shape_batch = multilik_batch_shape
        self.n_points = n_points
        self.decomp = decomp
        self.outputscales = outputscales

        if init_lmc_coeffs :
            U, S, V = compute_truncated_svd(Y=train_y, n_latents=n_latents)
            S = S / np.sqrt(n_points - 1) # seems to work better, but is it justified ?
            lmc_coefficients = (U * S.unsqueeze(-2)).mT
            self.variational_strategy.lmc_coefficients = torch.nn.Parameter(lmc_coefficients)  #shape (n_batch x) n_latents x n_tasks

    def forward( self, x:Tensor )-> Tensor:
        """
        Computes the prior distribution of the latent processes at the input locations. ! This does not return task-level values !
        Args:
            x: input data tensor
        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def lscales( self, unpacked:bool=True )-> Union[List[Tensor], Tensor]:  # returned shape : n_kernels x n_dims
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. Applies only if the model kernel is not composite

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_latents x n_dims (n_dim number of dimensions of the subkernel)
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
            unpacked: whether to trim useless dimensions of the tensor

        Returns:
            A tensors representing the learned outputscales of each subkernel and each task (shape n_latents x n_kernels)
        """
        n_kernels = len(self.covar_module.kernels) if hasattr(self.covar_module, 'kernels') else 1
        res = torch.zeros((self.n_latents, n_kernels))
        if n_kernels > 1:
            for i_ker in range(n_kernels):
                res[:, i_ker] = self.covar_module.kernels[i_ker].outputscale.data.squeeze()
        else:
            res[:, 0] = self.covar_module.outputscale.data.squeeze()
        return res.squeeze() if (n_kernels==1 and unpacked) else res
    
    def lmc_coefficients( self ) -> Tensor:
        """
        Returns the mixing matrix of the LMC model, which is a tensor of shape n_latents x n_tasks.
        Returns:
            A tensor of shape n_latents x n_tasks.
        """
        return self.variational_strategy.lmc_coefficients.data
    
    def compute_latent_distrib( self, x, prior=False, **kwargs):
        """
        Outputs (distributional) posterior values of the latent processes at the input locations.
        Args:
            x: input data tensor

        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        return self.variational_strategy.base_variational_strategy(x, prior=prior, **kwargs)
    
    def kernel_cond( self ) -> Tensor:
        """
        Computes the condition number of the inducing points kernel matrix.
        Returns:
            The condition number of the inducing points kernel matrix
        """
        with torch.no_grad():
            # Nice point : self.variational_strategy.base_variational_strategy.prior_distribution.lazy_covariance_matrix is cached by gpytorch
            K_plus = self.variational_strategy.base_variational_strategy.prior_distribution.lazy_covariance_matrix.evaluate()
        return torch.linalg.cond(K_plus)

    
    def save( self, extra_terms = False) -> dict:
        """
        Saves the model in a dictionary. The saved elements are strictly sufficient to make mean predictions (not variances).
        !! As of now, this method cannot accommodate : non-gaussian likelihoods, variable outputscales, nontrivial kernel decompositions,
        priors on kernel hyperparameters, and additional kernel settings (the ker_kwargs argument of the model). !!
        Args:
            extra_terms: whether to save terms of the model not needed for mean predictions, such as noise factors. Defaults to False.
        Returns:
            A dictionary containing the model's attributes.
        """
        dico = {}
        dico['kernel_type'] = self.covar_module.base_kernel.__class__.__name__ if self.outputscales is None else self.covar_module.__class__.__name__
        dico['mean_type'] = self.mean_module.__class__.__name__
        likelihood = self.likelihood
        with torch.no_grad():
            dico['lmc_coeffs'] = self.lmc_coefficients().tolist()
            if self.outputscales:
                dico['outputscales'] = self.outputscale().tolist()
            dico['lscales'] = self.lscales().tolist()
            inducing_points = self.variational_strategy.base_variational_strategy.inducing_points
            dico['inducing_points'] = inducing_points.tolist()
            with gp.settings.skip_posterior_variances(state=True), torch.no_grad():
                _ = self(torch.zeros_like(inducing_points)) # this is to compute the mean cache
            if not isinstance(self.variational_strategy.base_variational_strategy, gp.variational.UnwhitenedVariationalStrategy):
                warnings.warn('Model storage has only been tested for the UnwhitenedVariationalStrategy of gpytorch. \
                            The current strategy may not be supported.')
            dico['mean_cache'] = self.variational_strategy.base_variational_strategy._mean_cache.squeeze().tolist()

            if extra_terms:
                dico['noise_thresh'] = self.likelihood.raw_noise_constraint.lower_bound.item()
                noises = likelihood.noise if hasattr(likelihood, 'noise') else 0.
                if hasattr(likelihood, 'task_noises'):
                    noises = likelihood.task_noises + noises
                elif hasattr(likelihood, 'task_noise_covar'):
                    dico['task_noise_covar'] = likelihood.task_noise_covar.tolist()
                dico['noises'] = noises.tolist()
                if isinstance(self.variational_strategy.base_variational_strategy._variational_distribution, gp.variational.CholeskyVariationalDistribution):
                    dico['distrib_covar'] = self.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.tolist()
                else:
                    warnings.warn('The variational covariance was not stored, either deliberately (delta variational distribution) or because a nonstandard distribution was used.')
        return dico
    
    def default_mll(self):
        return gp.mlls.VariationalELBO(self.likelihood, self, num_data=self.n_points)



