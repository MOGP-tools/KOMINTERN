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

## making the mixing matrix a separate class allows to call torch.nn.utils.parametrizations.orthogonal
## onto it during instanciation of a ProjectedGPModel
class LMCMixingMatrix(torch.nn.Module):
    """
    Class for the parametrized mixing matrix of projected models. Making it a separate class allows to call 
    torch.nn.utils.parametrizations.orthogonal onto it during instanciation of a ProjectedGPModel
    """
    def __init__( self, Q_plus:Tensor, R:Tensor, bulk:bool=True ):
        """

        Args:
            Q_plus: (augmented) orthonormal part of the mixing matrix. See the reference article for explanations
            R: upper triangular part of the mixing matrix. See the reference article for explanations
        """
        super().__init__()
        if Q_plus.shape[1]==Q_plus.shape[0]:
            self.mode = 'Q_plus'
        elif Q_plus.shape[1]==R.shape[0]:
            self.mode = 'Q'
        else:
            raise ValueError('Wrong dimensions for Q_plus : should be n_tasks x n_tasks or n_tasks x n_latents')
        
        self.n_latents = R.shape[0]
        self.n_tasks = Q_plus.shape[0]
        self._size = torch.Size([self.n_latents, self.n_tasks])
        self.bulk = bulk
        if bulk:
            if self.mode=='Q_plus':
                R_padded = torch.eye(self.n_tasks)
                R_padded[:self.n_latents, :self.n_latents] = R
                H = Q_plus @ R_padded
            else:
                H = Q_plus @ R
            self.register_parameter("H", torch.nn.Parameter(H, requires_grad=True))
        else:
            self.register_parameter("Q_plus", torch.nn.Parameter(Q_plus, requires_grad=True))
            self.register_parameter("R", torch.nn.Parameter(R, requires_grad=True))

    def Q( self ):
        if self.mode=='Q_plus':
            return self.Q_plus[:, :self.n_latents]
        else:
            return self.Q_plus

    def Q_orth( self ):
        return self.Q_plus[:, self.n_latents:]

    def QR(self):
        if self.bulk:
            Q_plus, R_padded = torch.linalg.qr(self.H)
            if self.mode=='Q_plus':
                Q = Q_plus[:, :self.n_latents]
                Q_orth = Q_plus[:, self.n_latents:]
                R = R_padded[:self.n_latents, :self.n_latents]
            else:
                Q, Q_orth, R = Q_plus, None, R_padded
        else:
            Q, Q_orth, R = self.Q(), self.Q_orth(), self.R
        return Q, R, Q_orth

    def forward( self ):
        if self.bulk:
            if self.mode == 'Q':
                return self.H.T
            else:
                return self.H[:,:self.n_latents].T
        else:
            return (self.Q() @ self.R).T #format : n_latents x n_tasks

    def size( self, int=None ):
        if int:
            return self._size[int]
        else:
            return self._size


class ProjectedGPModel(ExactGPModel):
    """
    The projected LMC from the reference article.
    """
    def __init__( self, train_x:Tensor, train_y:Tensor, n_tasks:int, n_latents:int, proj_likelihood:Union[None,Likelihood]=None, 
                  init_lmc_coeffs:bool=False, BDN:bool=True, diagonal_B:bool=False, scalar_B:bool=False, diagonal_R:bool=False,
                  mean_type:Mean=gp.means.ConstantMean, ortho_param='matrix_exp', bulk=True,
                  noise_thresh:float=1e-4, noise_init:float=1e-2, outputscales:bool=False, eps=1e-3, **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel also apply here thanks to the inheritance.
        
        Args:
            train_x: training input data
            train_y: training input labels
            n_tasks: number of output tasks
            n_latents: number of latent processes
            proj_likelihood: batched independant likelihood of size n_latents for latent processes. Defaults to None.
            init_lmc_coeffs: whether to initialize LMC coefficients with SVD of the training labels. Defaults to False.
            BDN: whether to enforce the Block Diagonal Noise approximation (see reference article), making for a block-diagonal task noise matrix. Defaults to True.
            diagonal_B: whether to parametrize a diagonal noise factor B_tilde (see reference article), a simplification which theoretically causes no loss of generality. Defaults to False.
            scalar_B: whether to parametrize a scalar noise factor B_tilde (see reference article). Defaults to False.
            diagonal_R: whether to parametrize a diagonal scale component for the mixing matrix (see reference article). Defaults to False.
            mean_type: gp mean function for task-level processes. Defaults to gp.means.ConstantMean.
        """
        noise_thresh = np.log(noise_thresh)
        if proj_likelihood is None or proj_likelihood.noise.shape[0] != n_latents:
            warnings.warn("In projected GP model the dimension of the likelihood is the number of latent processes. "
                  "Provided likelihood was the wrong shape or None, so it was replaced by a fresh one")

            proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_latents]),
                                                    noise_constraint=gp.constraints.GreaterThan(np.exp(noise_thresh)))

        super().__init__(train_x, torch.zeros_like(train_y), proj_likelihood, n_tasks=n_latents, 
                         mean_type=gp.means.ZeroMean, outputscales=outputscales, **kwargs) # !! proj_likelihood will only be named likelihood in the model
        self.register_buffer('train_y', train_y)

        if mean_type is not gp.means.ZeroMean:
            raise ValueError('Projected GP model does not support non-zero output-wise means for now !')

        n_data, n_tasks = train_y.shape
        if init_lmc_coeffs:
            if scalar_B and BDN:
                Q_plus, R = init_lmc_coefficients(train_y, n_latents=n_latents, QR_form=True) # Q_plus has shape n_tasks x n_latents, R_padded has shape n_latents x n_latents        
            else:
                Q_plus, R_padded = init_lmc_coefficients(train_y, n_latents=n_tasks, QR_form=True) # Q_plus has shape n_tasks x n_tasks, R_padded has shape n_tasks x n_latents
                R = R_padded[:n_latents]
            # n_data, n_tasks = train_y.shape
            # if n_data > n_tasks:
            #     U, Sigma, VT = randomized_svd(train_y.cpu().numpy(), n_components=n_tasks, random_state=0)
            #     Q_plus, R = torch.as_tensor(VT.T, dtype=train_y.dtype), torch.as_tensor(np.diag(Sigma[:n_latents]) / np.sqrt(n_data - 1), dtype=train_y.dtype)
            # else:
            #     try:
            #         Q_plus, R_padded, Vt = torch.linalg.svd(train_y.T, full_matrices=True) 
            #         # we perform SVD rather than QR for the case where R must be diagonal (OILMM)
            #     except: # sklearn's randomized_svd is more stable than torch's svd in some cases
            #         U, Sigma, Vt = randomized_svd(train_y.cpu().numpy().T, n_components=n_data, random_state=0)
            #         Q_plus, R_padded = torch.as_tensor(U, dtype=train_y.dtype), torch.as_tensor(Sigma, dtype=train_y.dtype)
            #     if n_latents < n_data:
            #         R_padded = R_padded[:n_latents]
            #     else:
            #         R_padded, R_short = 1e-3*torch.ones(n_latents, dtype=train_y.dtype), R_padded
            #         R_padded[:n_data] = R_short
            #     R = torch.diag_embed(R_padded) / np.sqrt(n_data - 1)
        else:
            fake_coeffs = torch.randn(n_tasks, n_latents)
            Q_plus, R_padded, Vt = torch.linalg.svd(fake_coeffs) # Q_plus has shape n_tasks x n_tasks, R_padded has shape n_tasks x n_latents
            # we perform SVD rather than QR for the case where R must be diagonal (OILMM)
            R = R_padded[:n_latents]
            if scalar_B and BDN: # case of the PLMC_fast
                Q_plus = Q_plus[:,:n_latents]

        R = torch.diag_embed(R) / np.sqrt(n_data - 1)
        lmc_coefficients = LMCMixingMatrix(Q_plus, R, bulk=bulk)
        if not bulk:
            lmc_coefficients = torch.nn.utils.parametrizations.orthogonal(lmc_coefficients, name="Q_plus", orthogonal_map=ortho_param,
                                                                        use_trivialization=(ortho_param!='householder'))  # parametrizes Q_plus as orthogonal
            if diagonal_R:
                torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", PositiveDiagonalParam())
            else:
                torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", UpperTriangularParam())
        self.lmc_coefficients = lmc_coefficients

        if scalar_B:
            diagonal_B = True
            self.register_parameter("log_B_tilde", torch.nn.Parameter(np.log(noise_init) * torch.ones(n_tasks - n_latents)))
            torch.nn.utils.parametrize.register_parametrization(self, "log_B_tilde", ScalarParam(bounds=(noise_thresh, -noise_thresh)))
            if BDN:
                self.register_buffer('Y_squared_norm', (train_y**2).sum()) # case of the PLMC_fast (term for MLL computation)
        elif diagonal_B:
            self.register_parameter("log_B_tilde", torch.nn.Parameter(np.log(noise_init)*torch.ones(n_tasks - n_latents)))
            self.register_constraint("log_B_tilde", gp.constraints.GreaterThan(noise_thresh))
        else:
            self.register_parameter("B_tilde_inv_chol", torch.nn.Parameter(torch.diag_embed(np.log(1/noise_init)*torch.ones(n_tasks - n_latents))))
            torch.nn.utils.parametrize.register_parametrization(self, "B_tilde_inv_chol", LowerTriangularParam(bounds=(noise_thresh, -noise_thresh)))
        self.diagonal_B, self.scalar_B = diagonal_B, scalar_B

        if not BDN:
            self.register_parameter("M", torch.nn.Parameter(torch.zeros((n_latents, n_tasks - n_latents))))

        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.latent_dim = -1
        self.outputscales = outputscales
        self.eps = eps


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
        Q, R, Q_orth = self.lmc_coefficients.QR()
        H_pinv = torch.linalg.solve_triangular(R.T, Q, upper=False, left=False)  # shape n_tasks x n_latents
        if hasattr(self, "M"):
            return H_pinv + Q_orth @ self.M.T * self.projected_noise()[None,:]
        else:
            return H_pinv

    def project_data( self, data ):
        Q, R, Q_orth = self.lmc_coefficients.QR()
        unscaled_proj = Q.T @ data.T
        Hpinv_times_Y = torch.linalg.solve_triangular(R, unscaled_proj, upper=True)  # shape n_latents x n_points ; opposite convention to most other quantities !!
        if hasattr(self, "M"):
            return Hpinv_times_Y + self.projected_noise()[:,None] * self.M @ Q_orth.T @ data.T
        else:
            return Hpinv_times_Y

    def full_likelihood( self ):
        Q, R, Q_orth = self.lmc_coefficients.QR()
        res = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks, rank=self.n_tasks, has_global_noise=False)
        QR = Q @ R
        sigma_p = self.projected_noise()
        if sigma_p.is_cuda:
            res.cuda()
        if hasattr(self, "M"):
            if self.diagonal_B:
                B_tilde_root = torch.diag_embed(torch.exp(self.log_B_tilde / 2))
            else:
                B_tilde_root = torch.linalg.solve_triangular(self.B_tilde_inv_chol, 
                                            torch.eye(self.n_tasks - self.n_latents, device=self.B_tilde_inv_chol.device), upper=False).T
            B_tilde = B_tilde_root @ B_tilde_root.T
            B_term = Q_orth @ B_tilde @ Q_orth.T
            M_term = - QR @ (sigma_p[:,None] * self.M) @ B_tilde @ Q_orth.T
            Mt_term = M_term.T
            D_term_rotated = torch.diag_embed(sigma_p) + sigma_p[:,None] * self.M @ B_tilde @ self.M.T * sigma_p[None,:]
            D_term = QR @ D_term_rotated @ QR.T
        else:
            if self.scalar_B:
                if self.log_B_tilde.numel() > 0:
                    B_term = torch.exp(self.log_B_tilde[0]) * (torch.eye(self.n_tasks, device=self.log_B_tilde.device) - Q @ Q.T)
                else:
                    B_term = 0.
            else:
                if self.diagonal_B:
                    B_tilde_root = torch.diag_embed(torch.exp(self.log_B_tilde / 2))
                else:
                    B_tilde_root = torch.linalg.solve_triangular(self.B_tilde_inv_chol,
                        torch.eye(self.n_tasks - self.n_latents, device=self.B_tilde_inv_chol.device), upper=False).T
                B_term_root = Q_orth @ B_tilde_root
                B_term = B_term_root @ B_term_root.T
            M_term, Mt_term = 0., 0.
            D_term_root = QR * torch.sqrt(sigma_p)[None,:]
            D_term = D_term_root @ D_term_root.T

        Sigma = D_term + M_term + Mt_term + B_term
        # We use a while loop to ensure that the full noise covariance is positive definite.
        # We can deactivate gradient computation as loss computation does not involve the full likelihood
        with torch.no_grad(): 
            eps = 1e-6
            while eps < self.eps:
                try:
                    identity = torch.eye(self.n_tasks, dtype=res.task_noise_covar.dtype, device=res.task_noise_covar.device)
                    res.task_noise_covar_factor.data = torch.linalg.cholesky(Sigma + eps*identity)
                    break
                except:
                    eps *= 10
                    warnings.warn("Cholesky of the full noise covariance failed. Trying again with jitter {0} ...".format(eps))

        return res

    def B_tilde( self )-> Tensor:
        """
        Outputs the discarded noise factor B_tilde from the reference paper. 
        Returns:
            Discarded noise factor B_tilde (see reference paper), symmetric or even diagonal matrix of size (n_tasks - n_latents).
        """        
        if self.diagonal_B:
            return torch.diag_embed(torch.exp(self.log_B_tilde))
        else:
            L_inv = torch.linalg.solve_triangular(self.B_tilde_inv_chol, torch.eye(self.n_tasks - self.n_latents), upper=False)
            return L_inv.T @ L_inv

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
        proj_targets = self.project_data(self.train_y)
        super().set_train_data(inputs=self.train_inputs, targets=proj_targets, strict=False)
        batch_distrib = ExactGPModel.__call__(self, x, **kwargs)
        return batch_distrib  # shape n_latents x n_points
    
    def compute_loo(self, output=None, latent=False):
        train_x, train_y = self.train_inputs[0], self.train_y
        if output is None:
            output = self.compute_latent_distrib(train_x)
        K = self.likelihood(output).lazy_covariance_matrix
        y_proj = self.project_data(train_y).T
        identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
        L = K.cholesky(upper=False)
        loo_var = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
        loo_delta = L._cholesky_solve(y_proj.T.unsqueeze(-1), upper=False).squeeze(-1) * loo_var
        loo_var, loo_delta = loo_var.detach().T, loo_delta.detach().T
        if not latent:
            lmc_coeffs = self.lmc_coefficients()
            e_loo_raw = (loo_delta @ lmc_coeffs)
            diff = (self.train_y - self.project_data(self.train_y).T @ lmc_coeffs)
            loo_delta = e_loo_raw + diff
            loo_var = loo_var @ lmc_coeffs**2
        return loo_var, loo_delta

    
    def save( self):
        dico = {}
        Q, R, Q_orth = self.lmc_coefficients.QR()
        if self.lmc_coefficients.mode == 'Q_plus':
            dico['Q_orth'] = Q_orth.detach().tolist() 
        dico['Q'] = Q.detach().tolist()
        dico['R'] = R.detach().tolist()
        dico['Sigma_proj'] = self.projected_noise().detach().tolist()
        if self.diagonal_B:
            dico['Sigma_orth'] = torch.exp(self.log_B_tilde.detach()).tolist()
        else:
            dico['Sigma_orth'] = self.B_tilde_inv_chol.detach().tolist()
        if hasattr(self, 'M'):
            dico['M'] = self.M.detach().tolist()

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
        latent_covar = to_linear_operator(latent_covar.evaluate())
        covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
        covar = covar.add_jitter(self.eps)

        return gp.distributions.MultitaskMultivariateNormal(mean, covar)
    

class ProjectedLMCmll(gp.mlls.ExactMarginalLogLikelihood):
    """
    The loss function for the ProjectedGPModel. 
    """
    def __init__(self, latent_likelihood:Likelihood, model:ProjectedGPModel):
        """

        Args:
            latent_likelihood: the likelihood of a ProjectedGPModel (batched gaussian likelihood of size n_latents)
            model: any ProjectedGPModel.

        Raises:
            RuntimeError: rejects non-gaussian likelihoods.
        """        
        if not isinstance(latent_likelihood, gp.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ProjectedLMCmll, self).__init__(latent_likelihood, model)
        self.previous_lat = None


    def forward(self, latent_function_dist:gp.distributions.Distribution, target:Tensor, inputs=None, *params):
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
        proj_target = self.model.project_data(target) # shape n_latents x n_points

        # Get the log prob of the marginal distribution of latent processes
        latent_output = self.likelihood(latent_function_dist, *params) # shape n_latents x n_points
        latent_res = latent_output.log_prob(proj_target)
        latent_res = self._add_other_terms(latent_res, params).sum().div_(num_data)  # Scale by the amount of data we have

        # compute the part of likelihood lost by projection
        p, q = self.model.n_tasks, self.model.n_latents
        self.proj_term_list = [0]*3
        ## We store the additional terms in a list attribute in order to be able to plot them individually for testing
        Q, R, Q_orth = self.model.lmc_coefficients.QR()
        if not hasattr(self.model, 'M') and self.model.scalar_B:
            if self.model.log_B_tilde.numel() > 0:
                # log_B_tilde = torch.clamp(self.model.log_B_tilde, -9, 9)
                log_B_tilde = self.model.log_B_tilde
                B_tilde_inv_val = torch.exp(- log_B_tilde[0])
                log_B_tilde_root_diag = log_B_tilde / 2
                self.proj_term_list[1] = - 0.5 * B_tilde_inv_val * (self.model.Y_squared_norm - (target @ Q).pow(2).sum()).div_(num_data)
            else:
                self.proj_term_list[1] = 0.
                log_B_tilde_root_diag = torch.tensor([0.])
        else:
            if self.model.diagonal_B:
                log_B_tilde_root_diag = self.model.log_B_tilde / 2
                B_tilde_inv = torch.diag_embed(torch.exp(- self.model.log_B_tilde))
                rot_proj_target = target @ Q_orth
                discarded_noise_term = rot_proj_target @ B_tilde_inv @ rot_proj_target.T
            else:
                B_tilde_inv_root_diag = self.model.B_tilde_inv_chol[range(p-q), range(p-q)]
                log_B_tilde_root_diag = -torch.log(B_tilde_inv_root_diag)
                discarded_noise_root = target @ Q_orth @ self.model.B_tilde_inv_chol
                discarded_noise_term = discarded_noise_root @ discarded_noise_root.T
            self.proj_term_list[1] = - 0.5 * torch.trace(discarded_noise_term).div_(num_data)

        # All terms are implicitly or explicitly divided by the number of datapoints
        self.proj_term_list[0] = - 0.5 * 2 * torch.sum(log_B_tilde_root_diag) # factor 2 because of the use of a root
        if self.model.lmc_coefficients.bulk:
            self.proj_term_list[2] = - 0.5 * torch.log(R[range(q), range(q)]**2).sum()
        else:
            self.proj_term_list[2] = - 0.5 * 2 * self.model.lmc_coefficients.parametrizations.R.original[range(q), range(q)].sum()
        projection_term = sum(self.proj_term_list) - 0.5 * (p - q) * np.log(2*np.pi)

        res = latent_res + projection_term
        return res
