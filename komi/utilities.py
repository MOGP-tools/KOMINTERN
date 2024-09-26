from typing import Union, List
import math
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.kernels.kernel import Kernel
from sklearn.utils.extmath import randomized_svd

##----------------------------------------------------------------------------------------------------------------------
## Basics
def tensor_iter( *iterators ):
    if not iterators:
        yield ()
    else:
        for first in iterators[0]:
            for rest in tensor_iter(*iterators[1:]):
                yield (first,) + rest

## Library-independent metrics
def i_mean(x, dim=None):
    if isinstance(x, torch.Tensor):
        return torch.mean(x, dim=dim)
    else:
        return np.mean(x, axis=dim)
    
def i_log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    else:
        return np.log(x)
    
def i_sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    else:
        return np.sqrt(x)
def i_var(x, dim=None):
    if isinstance(x, torch.Tensor):
        return torch.var(x, dim=dim)
    else:
        return np.var(x, axis=dim)
    
def i_quantile(x, q):
    if isinstance(x, torch.Tensor):
        return torch.quantile(x, q)
    else:
        return np.quantile(x, q)

##----------------------------------------------------------------------------------------------------------------------

## Custom means and kernels

class SplineKernel(gp.kernels.Kernel):
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return (1 + x1**2 + x1**3 / 3).prod(dim=-1)
        mins = torch.min(x1.unsqueeze(-2), x2.unsqueeze(-3))
        maxes = torch.max(x1.unsqueeze(-2), x2.unsqueeze(-3))
        oned_vals = 1 + mins*maxes + 0.5 * mins**2 * (maxes - mins/3)
        res = oned_vals.prod(dim=-1)
        ## !! The following block is a workaround to handle batched inputs for nonstationary kernels. It should be removed when gpytorch is fixed
        if hasattr(self, 'batch_shape') and self.batch_shape and self.batch_shape != x1.shape[:-2]:
            batch_dim = -1 if params.get("last_dim_is_batch", False) else 0
            res = res.unsqueeze(batch_dim).expand(*self.batch_shape, *res.shape)
        return res

class PolynomialMean(gp.means.mean.Mean):
    def __init__( self, input_size, batch_shape=torch.Size(), bias=True, degree=3):
        super().__init__()
        for i in range(degree+1):
            self.register_parameter(name="weights_{0}".format(i),
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.degree = degree

    def forward( self, x: Tensor)-> Tensor:
        """

        Args:
            x: input data to be evaluated at

        Returns:
            A tensor of the values of the mean function at evaluation points.
        """
        res = 0
        for i in range(1, self.degree + 1):
            res += (x ** i).matmul(getattr(self, 'weights_{0}'.format(i))).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

class LinearMean(gp.means.mean.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    def basis_matrix( self, x ):
        return torch.hstack([x, torch.ones((len(x), 1), device=x.device)])

##----------------------------------------------------------------------------------------------------------------------
## Metrics and normalizations

def compute_macro_errs(errs, concs, keys, cc_words_idx=(0,1)):
    nonmacro_idx = [('macro' not in key) for key in keys]
    n_keys, n_errs = keys[nonmacro_idx], errs[:,nonmacro_idx]
    words = [key.split('_') for key in n_keys]
    cc_keys = ['_'.join([w[idx] for idx in cc_words_idx]) for w in words]
    concs_values = concs[cc_keys].values
    if type(errs) is torch.Tensor:
        concs_values = torch.as_tensor(concs_values, device=errs.device, dtype=errs.dtype)
    res = n_errs * concs_values
    return res


def transfo_mesh( array, return_coeffs=False, value=None, reverse=False):
    array = np.asarray(array)
    a, b = array[0], array[-1]
    if reverse:
        m, p = (b-a)/2, (a+b)/2
    else:
        m, p = 2 / (b - a), (a + b) / (a - b)
    if return_coeffs:
        return m, p
    if value is not None:
        return m * value + p
    else:
        return m * array + p


def max_norm_func( x: Tensor, axis: int = -1):
    return torch.max(torch.abs(x), dim=axis).values


class LeaveOneOutPseudoLikelihood(gp.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood):
    def __init__( self, likelihood, model, train_x, train_y):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model
        self.train_x = train_x
        self.train_y = train_y

    def forward( self, function_dist: gp.distributions.MultivariateNormal, target: Tensor, *params ) -> Tensor:
        output = self.likelihood(function_dist, *params)
        sigma2, target_minus_mu = self.model.compute_loo(output)
        term1 = -0.5 * sigma2.log()
        term2 = -0.5 * target_minus_mu.pow(2.0) / sigma2
        res = (term1 + term2).sum(dim=-1)

        res = self._add_other_terms(res, params)
        # Scale by the amount of data we have and then add on the scaled constant
        num_data = target.size(-1)
        return res.div_(num_data) - 0.5 * math.log(2 * math.pi)

##----------------------------------------------------------------------------------------------------------------------
    
## Model definition and initialization
    
def handle_covar_( kernel: Kernel, dim: int, decomp: Union[List[List[int]], None]=None, n_funcs:int=1,
                   prior_scales:Union[Tensor,None]=None, prior_width:Union[Tensor,None]=None, outputscales:bool=True,
                   ker_kwargs:Union[dict, None]=None )-> Kernel:

    """ An utilitary to create and initialize covariance functions.

    Args:
        kernel: basis kernel type
        dim: dimension of the data (number of variables)
        decomp: instructions to create a composite kernel with subgroups of variables. Defaults to None
        Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2)
        n_funcs: batch dimension (number of tasks or latent functions depending on the case), defaults to 1
        prior_scales: mean values of the prior for characteristic lengthscales. Defaults to None
        prior_width: deviation_to_mean ratio of the prior for characteristic lengthscales. Defaults to None
        outputscales: whether or not the full kernel has a learned scaling factor, i.e k(x) = a* k'(x). 
        If decomp is nontrivial, each subkernel is automatically granted an outputscale. Defaults to True
        ker_kwargs: additional arguments to pass to the underlying gp kernel. Defaults to None

    Returns:
        A gp-compatible kernel
    """
    if ker_kwargs is None:
        ker_kwargs = {}

    if decomp is None:
        decomp = [list(range(dim))]

    l_priors = [None] * len(decomp)
    if prior_scales is not None:
        if prior_width is None:
            raise ValueError('A prior width should be provided if a prior mean is')
        if type(prior_scales) is not list:  # 2 possible formats: an array with one length per variable, or a list with one array per kernel
            prior_scales = [prior_scales[idx_list] for idx_list in decomp]
        if type(prior_width) is not list:   # 2 possible formats: an array with one length per variable, or a list with one array per kernel
            prior_width = [prior_width[idx_list] for idx_list in decomp]

        for i_ker, idx_list in enumerate(decomp):
            if len(idx_list) > 1:
                l_priors[i_ker] = gp.priors.MultivariateNormalPrior(loc=prior_scales[i_ker],
                                            covariance_matrix=torch.diag_embed(prior_scales[i_ker]*prior_width[i_ker]))
            else:
                l_priors[i_ker] = gp.priors.NormalPrior(loc=prior_scales[i_ker],
                                                              scale=prior_scales[i_ker]*prior_width[i_ker])

    kernels_args = [{'ard_num_dims': len(idx_list), 'active_dims': idx_list, 'lengthscale_prior': l_priors[i_ker],
                         'batch_shape': torch.Size([n_funcs])} for i_ker, idx_list in enumerate(decomp)]

    kernels = []
    for i_ker, ker_args in enumerate(kernels_args):
        ker = kernel(**ker_args, **ker_kwargs)
        kernels.append(ker)

    if len(decomp) > 1 :
        covar_module = gp.kernels.ScaleKernel(kernels[0], batch_shape=torch.Size([n_funcs]))
        for ker in kernels[1:]:
            covar_module += gp.kernels.ScaleKernel(ker, batch_shape=torch.Size([n_funcs]))
    else:
        if outputscales:
            covar_module = gp.kernels.ScaleKernel(kernels[0], batch_shape=torch.Size([n_funcs]))
        else:
            covar_module = kernels[0]

    if prior_scales is not None and kernels[0].has_lengthscale:
        try:
            if len(decomp) > 1 :
                for i_ker in range(len(kernels)):
                        covar_module.kernels[i_ker].base_kernel.lengthscale = l_priors[i_ker].mean
            elif outputscales:
                covar_module.base_kernel.lengthscale = l_priors[0].mean
            else:
                covar_module.lengthscale = l_priors[0].mean
        except:
            raise ValueError('Provided prior scales were of the wrong shape')

    return covar_module

def init_lmc_coefficients( train_y: Tensor, n_latents: int, QR_form:bool=False):
    n_data, __ = train_y.shape
    if n_data >= n_latents:
        U, S, Vt = randomized_svd(train_y.cpu().numpy().T, n_components=n_latents, random_state=0)
        U, S = torch.as_tensor(U, device=train_y.device, dtype=train_y.dtype), torch.as_tensor(S, device=train_y.device, dtype=train_y.dtype)
    else:
        Q, R = np.linalg.qr(train_y.cpu().numpy().T, mode='complete')
        S = 1e-3 * torch.ones(n_latents, device=train_y.device, dtype=train_y.dtype)
        S[:n_data] = torch.as_tensor(np.diag(R).copy(), device=train_y.device, dtype=train_y.dtype)
        U = torch.as_tensor(Q[:,:n_latents], device=train_y.device, dtype=train_y.dtype)
    S = S / np.sqrt(n_data - 1)
    if QR_form:
        return U, S
    else:
        y_transformed = U * S
    return y_transformed.T
##----------------------------------------------------------------------------------------------------------------------

## Parametrizations

class ScalarParam(torch.nn.Module):
    """
    Torch parametrization for a scalar matrix.
    """
    def __init__( self, bounds: List[float] = [1e-16, 1e16]):
        super().__init__()
        self.bounds = bounds

    def forward( self, X :Tensor)-> Tensor:
        return torch.ones_like(X) * torch.clamp(X.mean(), *self.bounds)
    def right_inverse( self, A :Tensor)-> Tensor:
        return A

class PositiveDiagonalParam(torch.nn.Module):
    """
    Torch parametrization for a positive diagonal matrix.
    """
    def forward( self, X: Tensor)-> Tensor:
        return torch.diag_embed(torch.exp(torch.diag(X)))
    def right_inverse( self, A: Tensor)-> Tensor:
        return torch.diag_embed(torch.log(torch.diag(A)))

class UpperTriangularParam(torch.nn.Module):
    """
    Torch parametrization for an upper triangular matrix.
    """
    def forward( self, X: Tensor)-> Tensor:
        upper =  X.triu()
        upper[range(len(upper)), range(len(upper))] = torch.exp(upper[range(len(upper)), range(len(upper))])
        return upper
    def right_inverse( self, A: Tensor)-> Tensor: 
        res = A
        res[range(len(res)), range(len(res))] = torch.log(res[range(len(res)), range(len(res))])
        return res

class LowerTriangularParam(torch.nn.Module):
    """
    Torch parametrization for a Cholesky factor matrix (lower triangular with positive diagonal).
    """
    def __init__(self, bounds: List[float] = [1e-16, 1e16]):
        super().__init__()
        self.bounds = bounds

    def forward( self, X: Tensor )-> Tensor:
        lower = X.tril()
        lower[range(len(lower)), range(len(lower))] = torch.exp(torch.clamp(lower[range(len(lower)), range(len(lower))], *self.bounds))
        return lower
    def right_inverse( self, A: Tensor)-> Tensor:
        res = A
        res[range(len(res)), range(len(res))] = torch.log(res[range(len(res)), range(len(res))])
        return res