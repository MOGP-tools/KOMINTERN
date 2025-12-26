from functools import reduce #, lru_cache
from typing import Union, List, Tuple
import warnings
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.means.mean import Mean
from gpytorch.likelihoods.likelihood import Likelihood
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.operators.dense_linear_operator import to_linear_operator

from .utilities import init_lmc_coefficients, compute_truncated_svd, \
    ScalarParam, PositiveDiagonalParam, LowerTriangularParam, UpperTriangularParam
from .base_gp import ExactGPModel

## making the mixing matrix a separate class allows to call torch.nn.utils.parametrizations.orthogonal
## onto it during instanciation of a ProjectedGPModel
class FastMixingMatrix(torch.nn.Module):
    """
    Class for the parametrized mixing matrix of projected models. Making it a separate class allows to call 
    torch.nn.utils.parametrizations.orthogonal onto it during instanciation of a ProjectedGPModel
    """
    def __init__( self, Q:Tensor, R:Tensor):
        """
        Args:
            Q: orthonormal part of the mixing matrix, of shape n_tasks x n_latents
            R: upper triangular part of the mixing matrix, of shape n_latents x n_latents
        """
        super().__init__()
        if len(Q.shape) != len(R.shape):
            raise ValueError("Q and R have different number of axes: {0} and {1}".format(Q.shape, R.shape))
        else:
            self.shape_batch = Q.shape[:-2]
            self.n_batch_dims = len(self.shape_batch)

        if Q.shape[self.n_batch_dims + 1] != R.shape[self.n_batch_dims + 0]:
            raise ValueError('Wrong dimensions for Q : should be (n_batch x) n_tasks x n_latents,' \
            'got {0}. n_latents has been infered from R to be {1}'.format(Q.shape, R.shape[self.n_batch_dims + 0]))
        
        self.n_latents = R.shape[self.n_batch_dims]
        self.n_tasks = Q.shape[self.n_batch_dims]
        self._size = torch.Size([*self.shape_batch, self.n_latents, self.n_tasks])
        H = Q @ R
        self.register_parameter("H", torch.nn.Parameter(H, requires_grad=True))

    def QR(self) -> Tuple[Tensor, Tensor]:
        """
        Outputs the Q and R factors of the mixing matrix
        Returns:
            Q factor of the mixing matrix, of shape n_tasks x n_latents.
            R factor of the mixing matrix, of shape n_latents x n_latents.
        """
        Q, R_padded = torch.linalg.qr(self.H)
        Q, R = Q, R_padded
        return Q, R

    def forward( self ) -> Tensor:
        """
        Outputs the full (batch) mixing matrix H, in transposed form in order to match the standard storage format of data labels.
        Returns:
            Transposed mixing matrix H, of shape (n_batch x) n_tasks x n_latents.
        """
        return self.H.mT

    def size( self, int=None ) -> Union[int, torch.Size]:
        if int:
            return self._size[int]
        else:
            return self._size


class FastProjectedGPModel(ExactGPModel):
    """
    The projected LMC model. Reference : https://arxiv.org/abs/2310.12032
    """
    def __init__( self,
                  train_x:Tensor,
                  train_y:Tensor,
                  n_latents:int,
                  proj_likelihood:Union[None,Likelihood]=None, 
                  mean_type:Mean=gp.means.ZeroMean,
                  noise_thresh:float=1e-4,
                  outputscales:bool=False,
                  jitter_val:Union[float,None]=None,
                  **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel (in particular the choice of 
        mean and kernel function) also apply here thanks to the inheritance.
        
        Args:
            train_x: training input data
            train_y: training input labels
            n_latents: number of latent processes
            proj_likelihood: batched independant likelihood of size n_latents for latent processes. Defaults to None.
            mean_type: gp mean function for task-level processes. At the moment, only a zero mean is implemented ; every other choice will throw an error.
            Defaults to gp.means.ZeroMean.
            noise_thresh: minimum value for the noise parameter. Has a large impact for ill-conditioned kernel matrices, which is the case of the HXS application. Defaults to 1e-6.
            outputscales: whether to endow each latent kernel with a learned scaling factor, k(.) = a*k_base(.). This is only useful for predictive variance 
            scaling, and may result in over-parametrization. Defaults to False
            jitter_val: jitter value for the Cholesky decomposition of the full noise covariance matrix, and for addition to the predictive covariance matrix.
            If None, it is set to the default gpytorch Cholesky jitter setting. Defaults to None.
        """
        if mean_type is not gp.means.ZeroMean:
            raise NotImplementedError('Projected GP model does not support non-zero output-wise means for now !')

        if len(train_y.shape) == 2:
            n_points, n_tasks = train_y.shape
            axes_layout = {'n_points':0, 'n_tasks':1}
            latent_batch_shape = torch.Size([n_latents])
            discarded_noise_shape = torch.Size([n_tasks - n_latents])
            batch_shape = torch.Size()
        elif len(train_y.shape) == 3:
            n_batch, n_points, n_tasks = train_y.shape
            axes_layout = {'n_batch':0, 'n_points':1, 'n_tasks':2}
            latent_batch_shape = torch.Size([n_batch, n_latents])
            discarded_noise_shape = torch.Size([n_batch, n_tasks - n_latents])
            batch_shape = torch.Size([n_batch])

        # Likelihood (noise model) initialization
        noise_init = 10 * noise_thresh
        if proj_likelihood is not None and proj_likelihood.noise.shape[-1] != n_latents:
            raise ValueError("In projected GP model the dimension of the likelihood is the number of latent processes. "
                  "Provided likelihood has length {0} while n_latents is {1}".format(proj_likelihood.noise.shape[-1], n_latents))
        elif proj_likelihood is None:
            proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=latent_batch_shape,
                                        noise_constraint=gp.constraints.GreaterThan(noise_thresh))
            proj_likelihood.noise = noise_init * torch.ones_like(proj_likelihood.noise)
            
        # Initialization of LMC coefficients and projected data
        U, S, V = compute_truncated_svd(Y=train_y, n_latents=n_latents)
        R = S
        Q = U
        proj_y = V.mT
        R = torch.diag_embed(R)
        lmc_coefficients = FastMixingMatrix(Q, R)

        # Initialization of the latent processes
        super().__init__(train_x=train_x, train_y=proj_y, likelihood=proj_likelihood,
                         mean_type=gp.means.ZeroMean, outputscales=outputscales, batch_lik=True, **kwargs)
        # !! proj_likelihood will only be named likelihood in the model
        self.register_buffer('train_y', train_y)
        self.lmc_coefficients = lmc_coefficients

        # Initialization of the discarded noise terms ; see PLMC article
        discarded_noise_tens = torch.ones(discarded_noise_shape)
        log_noise_thresh = np.log(noise_thresh)
        log_init_noise = np.log(noise_init)
        self.register_parameter("log_B_tilde", torch.nn.Parameter(log_init_noise * discarded_noise_tens))
        torch.nn.utils.parametrize.register_parametrization(self, "log_B_tilde", ScalarParam(bounds=(log_noise_thresh, -log_noise_thresh)))
        self.register_buffer('Y_squared_norm', (train_y**2).sum()) # case of the PLMC_fast (term for MLL computation)

        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.shape_batch = batch_shape
        self.latent_dim = -1
        self.outputscales = outputscales
        if jitter_val is None:
            self.jitter_val = gp.settings.cholesky_jitter.value(train_x.dtype)
        else:
            self.jitter_val = jitter_val


    def projected_noise( self )-> Tensor:
        """
        Returns a vector containing the modeled noises of latent processes. Its diagonal embedding is the matrix Sigma_P from the article.
        Returns:
            Modeled noise vector of size (n_batch x) n_latents. 
        """
        return self.likelihood.noise.squeeze(-1)
    
    # @lru_cache(maxsize=None) # caching projected data and projected matrix is appealing, but it messes with backpropagation. No workaround has been found yet
    def projection_matrix( self )-> Tensor:
        """
        Returns matrix T from the reference article, such that YT is the "projected data" seen by latent processes
        Returns:
            Projection matrix T, of shape (n_batch x) n_tasks x n_latents. 
        """
        Q, R = self.lmc_coefficients.QR()
        H_pinv = torch.linalg.solve_triangular(R.mT, Q, upper=False, left=False)  # shape (n_batch x) n_tasks x n_latents
        return H_pinv

    def project_data( self, data ) -> Tensor:
        """
        Projects some data labels onto the latent space.
        Args:
            data: data tensor of shape (n_batch x) n_points x n_tasks
        Returns:
            Projected data tensor of shape (n_batch x) n_latents x n_points.
            This shape convention corresponds to the batch treatment in gpytorch, not to the usual convention.
        """
        Q, R = self.lmc_coefficients.QR()
        unscaled_proj = Q.mT @ data.mT
        Hpinv_times_Y = torch.linalg.solve_triangular(R, unscaled_proj, upper=True)  
        return Hpinv_times_Y # (n_batch x) shape n_latents x n_points ; opposite convention to most other quantities !!

    def full_likelihood( self, diag=False ) -> gp.likelihoods.MultitaskGaussianLikelihood:
        """
        Outputs the task-level likelihood of the model (Sigma matrix from the reference article), including the noise of the latent processes and the discarded noise.
        Returns:
            Task-level likelihood of the model, with a multitask gaussian likelihood of size n_tasks.
        """
        Q, R = self.lmc_coefficients.QR()
        QR = Q @ R
        sigma_p = self.projected_noise().unsqueeze(-2)
        discarded_noise_size = self.n_tasks - self.n_latents
        if self.n_latents < self.n_tasks:
            B_tilde = torch.exp(self.log_B_tilde[..., :1])
            if diag:
                B_term = B_tilde * (1 - (Q**2).sum(dim=-1))
            else:
                identities = torch.broadcast_to(
                    torch.eye(self.n_tasks, device=self.log_B_tilde.device),
                    (*self.shape_batch, self.n_tasks, self.n_tasks))
                B_term = torch.broadcast_to(B_tilde.unsqueeze(-1), identities.shape) * (identities - Q @ Q.mT)
        else:
            B_term = 0.

        D_term_root = QR * torch.sqrt(sigma_p)
        D_term = D_term_root @ D_term_root.mT if not diag else (D_term_root**2).sum(dim=-1)

        if diag:
            res = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks, batch_shape=self.shape_batch,
                                                             rank=0, has_global_noise=False)
            if sigma_p.is_cuda:
                res.cuda()
            res.task_noises = B_term + D_term
        else:
            res = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks, batch_shape=self.shape_batch,
                                                             rank=self.n_tasks, has_global_noise=False)
            if sigma_p.is_cuda:
                res.cuda()
            Sigma = D_term + B_term
            # We use a while loop to ensure that the full noise covariance is positive definite.
            # We can deactivate gradient computation as loss computation does not involve the full likelihood
            with torch.no_grad(): 
                eps = self.jitter_val
                while eps < 1e6 * self.jitter_val:
                    try:
                        identities = torch.broadcast_to(torch.eye(self.n_tasks, dtype=res.task_noise_covar.dtype,
                                        device=res.task_noise_covar.device), (*self.shape_batch, self.n_tasks, self.n_tasks))
                        res.task_noise_covar_factor.data = torch.linalg.cholesky(Sigma + eps*identities)
                        break
                    except:
                        eps *= 10
                        warnings.warn("Cholesky of the full noise covariance failed. Trying again with jitter {0} ...".format(eps))
        return res

    def B_tilde( self )-> Tensor:
        """
        Outputs the discarded noise factor B_tilde from the reference paper. 
        Returns:
            Discarded noise factor B_tilde (see reference paper), symmetric or diagonal matrix of size (n_tasks - n_latents).
        """        
        return torch.diag_embed(torch.exp(self.log_B_tilde))

    def forward( self, x:Tensor )-> gp.distributions.MultivariateNormal:  # ! forward only returns values of the latent processes !
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

    def compute_latent_distrib( self, x:Tensor, **kwargs )-> gp.distributions.MultivariateNormal:
        """
        Outputs (distributional) posterior values of the latent processes at the input locations. This is the function which is called to compute
        the loss during training.
        Args:
            x: input data tensor

        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        proj_targets = self.project_data(self.train_y)
        super().set_train_data(inputs=self.train_inputs, targets=proj_targets, strict=False)
        batch_distrib = ExactGPModel.__call__(self, x, **kwargs)
        return batch_distrib  # shape (n_batch x) n_latents x n_points
    
    def compute_loo(self, output=None, latent=False) -> Tuple[Tensor, Tensor]:
        """
        Computes the leave-one-out (LOO) variance and error gaps (y_true - y_loo) values for the model.
        Args:
            output: the latent distribution of the model at the training points. If None, it is computed.
            latent: whether to compute the leave-one-out errors at the latent level (True) or at the task level (False). Default is False.
            train_y: the training labels. If None and latent=False, they must be stored in the model. Default is None.
        Returns:
            A tuple containing the LOO variances and error gaps for each task (each of size n_points x n_tasks, or n_points x n_latents if latent=True).
        """
        # TODO : adapt to batch case
        train_x, train_y = self.train_inputs[0], self.train_y
        with torch.no_grad():
            if output is None:
                output = self.compute_latent_distrib(train_x)
            K = self.likelihood(output).lazy_covariance_matrix
            y_proj = self.project_data(train_y)
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            L = K.cholesky(upper=False)
            loo_var = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
            loo_delta = L._cholesky_solve(y_proj.unsqueeze(-1), upper=False).squeeze(-1) * loo_var
            loo_var, loo_delta = loo_var.mT, loo_delta.mT
            if not latent:
                lmc_coeffs = self.lmc_coefficients()
                e_loo_raw = (loo_delta @ lmc_coeffs)
                diff = (self.train_y - y_proj.mT @ lmc_coeffs)
                loo_delta = e_loo_raw + diff
                loo_var = loo_var @ lmc_coeffs**2
        return loo_var, loo_delta


    def set_train_data( self, inputs:Tensor, targets:Tensor, strict:bool=True ):
        """
        Replaces the current training data of the model. Overrides the parent method to store the training labels in the model.
        """
        super().set_train_data(inputs=inputs, targets=self.project_data(targets), strict=strict)
        self.train_y = targets

    
    def save( self, extra_terms=False) -> dict:
        """
        Saves the model in a dictionary. The saved elements are strictly sufficient to make mean predictions (not variances).
        !! As of now, this method cannot accommodate : non-gaussian likelihoods, variable outputscales, nontrivial kernel decompositions,
        priors on kernel hyperparameters, and additional kernel settings (the ker_kwargs argument of the model). !!
        Args:
            extra_terms: whether to save terms of the model not needed for mean predictions, such as noise factors. Defaults to False.
        Returns:
            A dictionary containing the model's attributes.
        """
        self.eval()
        dico = {}
        dico['kernel_type'] = self.covar_module.base_kernel.__class__.__name__ if self.outputscales is None else self.covar_module.__class__.__name__
        with torch.no_grad():
            if extra_terms:
                dico['noise_thresh'] = self.likelihood.raw_noise_constraint.lower_bound.item()
                dico['diagonal_B'] = self.diagonal_B
                dico['scalar_B'] = self.scalar_B
                dico['diagonal_R'] = self.diagonal_R
                Q, R = self.lmc_coefficients.QR()
                dico['Q'] = Q.tolist()
                dico['R'] = R.tolist()
                dico['Sigma_proj'] = self.projected_noise().tolist()
                if self.diagonal_B:
                    dico['Sigma_orth'] = torch.exp(self.log_B_tilde).tolist()
                else:
                    dico['Sigma_orth'] = self.B_tilde_inv_chol.tolist()
                if hasattr(self, 'M'):
                    dico['M'] = self.M.tolist()
            else:
                dico['lmc_coeffs'] = self.lmc_coefficients().tolist()

            with torch.no_grad():
                _ = self(torch.zeros_like(self.train_inputs[0])) # this is to compute the mean cache
            dico['mean_cache'] = self.prediction_strategy.mean_cache.tolist()
            dico['lscales'] = self.lscales().tolist()
            if self.outputscales:
                dico['outputscales'] = self.outputscales().tolist()
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
        
        super().set_train_data(inputs=self.train_inputs, targets=self.project_data(self.train_y), strict=False)
        latent_dist = ExactGPModel.__call__(self, x, **kwargs)

        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        num_dim = num_batch + len(latent_dist.event_shape)
        lmc_coefficients = self.lmc_coefficients().expand(*latent_dist.batch_shape, self.lmc_coefficients.size(-1))

        # Mean: ... x N x n_tasks
        latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
        mean = latent_mean @ lmc_coefficients.permute(
            *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
        )

        # Covar: ... x (N x n_tasks) x (N x n_tasks)
        latent_covar = latent_dist.lazy_covariance_matrix
        lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
        # latent_covar = to_linear_operator(latent_covar.evaluate())
        covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
        covar = covar.add_jitter(self.jitter_val)

        return gp.distributions.MultitaskMultivariateNormal(mean, covar)
    
    def default_mll(self):
        return FastProjectedLMCmll(self.likelihood, self)
    

class FastProjectedLMCmll(gp.mlls.ExactMarginalLogLikelihood):
    """
    The loss function for the FastProjectedGPModel. 
    """
    def __init__(self, latent_likelihood:Likelihood, model:FastProjectedGPModel):
        """

        Args:
            latent_likelihood: the likelihood of a FastProjectedGPModel (batched gaussian likelihood of size n_latents)
            model: any FastProjectedGPModel.

        Raises:
            RuntimeError: rejects non-gaussian likelihoods.
        """        
        if not isinstance(latent_likelihood, gp.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(FastProjectedLMCmll, self).__init__(latent_likelihood, model)
        self.previous_lat = None


    def forward(self, latent_function_dist:gp.distributions.Distribution, target:Tensor, inputs=None, *params) -> Tensor:
        """
        Computes the value of the loss (MLL) given the model predictions and the observed values at training locations. 
        Args:
            latent_function_dist: gp batched gaussian distribution of size n_latents x n_points representing the values of latent processes.
            target: training labels Y of shape n_points x n_tasks

        Raises:
            RuntimeError: rejects non-gaussian latent distributions.

        Returns:
            The (scalar) value of the MLL loss for this model and data.
        """        
        if not isinstance(latent_function_dist, gp.distributions.multivariate_normal.MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        num_data = latent_function_dist.event_shape.numel()
        
        # project the targets
        proj_target = self.model.project_data(target) # shape (n_batch x) n_latents x n_points

        # Get the log prob of the marginal distribution of latent processes
        latent_output = self.likelihood(latent_function_dist, *params) # shape (n_batch x) n_latents x n_points
        latent_res = latent_output.log_prob(proj_target)
        latent_res = self._add_other_terms(latent_res, params).sum().div_(num_data)  # Scale by the amount of data we have

        # compute the part of likelihood lost by projection
        p, q = self.model.n_tasks, self.model.n_latents
        self.proj_term_list = [0]*3
        ## We store the additional terms in a list attribute in order to be able to plot them individually for testing
        Q, R = self.model.lmc_coefficients.QR()
        if self.model.log_B_tilde.numel() > 0:
            # log_B_tilde = torch.clamp(self.model.log_B_tilde, -9, 9)
            log_B_tilde = self.model.log_B_tilde
            B_tilde_inv_val = torch.exp(- log_B_tilde[0])
            log_B_tilde_root_diag = log_B_tilde / 2
            self.proj_term_list[1] = B_tilde_inv_val * (self.model.Y_squared_norm - (target @ Q).pow(2).sum()).div_(num_data)
            # the parenthesis is the squared norm of the projection of target onto the space orthogonal to span(Q)
        else:
            self.proj_term_list[1] = 0.
            log_B_tilde_root_diag = torch.tensor([0.])

        # All terms are implicitly or explicitly divided by the number of datapoints
        self.proj_term_list[0] = 2 * torch.sum(log_B_tilde_root_diag) # factor 2 because of the use of a root
        self.proj_term_list[2] = torch.log(R[..., range(q), range(q)]**2).sum() # keep the square in the log because the quantity can be negative
        projection_term = sum(self.proj_term_list) + (p - q) * np.log(2*np.pi)

        res = latent_res - 0.5 * projection_term
        return res
