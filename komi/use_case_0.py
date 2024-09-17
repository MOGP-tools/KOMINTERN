import pickle
import json
import zipfile

import os
num_threads = 20
os.environ["OMP_NUM_THREADS"] = str(num_threads) # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(num_threads) # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads) # export NUMEXPR_NUM_THREADS=6

import numpy as np
np.random.seed(seed=12)
import pandas as pd

import torch
torch.manual_seed(12)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float32)

import gpytorch as gp
from komi.utilities import transfo_mesh, SplineKernel, LinearMean, PolynomialMean
from mogp_lazy import LazyLMCModel
from mogp_var import VariationalMultitaskGPModel
from mogp_plmc import ProjectedGPModel, ProjectedLMCmll
from mogp_icm import MultitaskGPModel
from train_gp import train_model, eval_model

# from custom_profiler import profiler, magic_profiler # info : https://github.com/KarGeekrie/customProfiler

def cround(n, ndec=2):
    if n == 0:
        return n
    else:
        d = np.ceil(-np.log10(abs(n))).astype(int)
        return np.round(n, d + ndec)

met_dict = {
    'alpha_CI': lambda rwm : torch.mean((rwm['errs'] < 2 * rwm['sigmas']).float()),
    'PVA': lambda rwm : torch.log(torch.mean(rwm['errs2'] / rwm['vars'], dim=0)).mean(),
    'R2': lambda rwm : (1 - torch.mean(rwm['errs2'], dim=0) / torch.var(rwm['y_test'], dim=0)).mean(),
    'RMSE': lambda rwm : torch.sqrt(rwm['errs2'].mean()),
    'mean_err_abs': lambda rwm : rwm['errs'].mean(),
    'max_err_abs': lambda rwm : rwm['errs'].max(),
    'mean_err_quant05': lambda rwm : torch.quantile(rwm['errs'], 0.05),
    'mean_err_quant95': lambda rwm : torch.quantile(rwm['errs'], 0.95),
    'mean_err_quant99': lambda rwm : torch.quantile(rwm['errs'], 0.99),
}

print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
ndec = 2
compute_loo = True
make_preds = True
export_preds = False
export_data_to_np = False
save_torch_model = False
file_format = 'json' # 'json' or 'pickle'

if __name__ == "__main__":

    ## Data preparation

    # assembly_name = ?
    # disc = 'FA'
    # ngroup = 'grp02'
    # mix_name = 'L_chain'
    # hom_name = ngroup + '_ENE_' + disc + '_' + mix_name
    # variables = ['root_bu', 'tf', 'tm', 'br']
    # vars_n = [var + '_n' for var in variables] #normalized variables
    # all_vars = variables + vars_n + ['bu']

    # min_bu, max_bu = 75., 62000.
    # var_ranges = {'root_bu': [np.sqrt(min_bu), np.sqrt(max_bu)], 'tf': [273.15, 2073.15], 'tm':[273.15, 600.], 'br': [0.,2500.]}

    # train_df = pd.read_csv(???, index_col=0)
    # test_df = pd.read_csv(???, index_col=0)

    # if ngroup == 'grp20' or mix_name in ['L_chain', 'XL_chain']: #elimintation of almost-zero HXS
    #     filt = (np.abs(train_df).mean(axis=0) > 1e-5)
    #     train_df = train_df.loc[:, filt]

    # mask = (train_df['bu']>min_bu)&(train_df['bu']<max_bu) #elimination of points with bu outside the range
    # train_df = train_df[mask]
    # mask = (test_df['bu']>min_bu)&(test_df['bu']<max_bu)
    # test_df = test_df[mask]
 
    # xs_keys = train_df.columns.difference(all_vars) #just HXS, no variables
    # Y = torch.as_tensor(train_df[xs_keys].values)
    # Y_test = torch.as_tensor(test_df[xs_keys].values)
    # means, maxes = Y.mean(dim=0), Y.abs().max(dim=0)
    # Y = (Y - means) / maxes
    # Y_test = (Y_test - means) / maxes

    # train_df['root_bu'] = np.sqrt(train_df['bu'].values)
    # test_df['root_bu'] = np.sqrt(test_df['bu'].values)
    # train_x = train_df[variables].copy()
    # test_x = test_df[variables].copy()
    # for var in variables:
    #     train_x[var + '_n'] = transfo_mesh(var_ranges[var], value=train_x[var])
    #     test_x[var + '_n'] = transfo_mesh(var_ranges[var], value=test_x[var])
    # X = torch.as_tensor(train_x[vars_n].values)
    # X_test = torch.as_tensor(test_x[vars_n].values)

    root = '../useCase/neutro_data/'
    X = torch.load(root + 'train_x_sobol256.pt', weights_only=True)
    Y = torch.load(root + 'train_data_02g_FA_Lchain.pt', weights_only=True)
    if make_preds:
        X_test = torch.load(root + 'test_x_LHS512.pt', weights_only=True)
        Y_test = torch.load(root + 'test_data_02g_FA_Lchain.pt', weights_only=True)
    means = torch.load(root + 'means_02g_FA_Lchain.pt', weights_only=True)
    maxes = torch.load(root + 'maxes_02g_FA_Lchain.pt', weights_only=True)
    xs_keys = pickle.load(open(root + 'keys_02g_FA_Lchain.pickle', 'rb'))
    # X, X_test, Y, Y_test = X[::2], X_test[::2], Y[::2, :30], Y_test[::2, :30]

    n_points, n_tasks = Y.shape
    case = 'b0py26e_02g_FA'  # enter the name of the run here
    exproot = '../results/'
    if export_data_to_np:
        np.save(exproot + 'train_x_sobol256_full.npy', X.numpy())
        np.save(exproot + 'train_data_02g_FA_Lchain_full.npy', Y.numpy())
        if make_preds:
            np.save(exproot + 'test_x_LHS512_full.npy', X_test.numpy())
            np.save(exproot + 'test_data_02g_FA_Lchain_full.npy', Y_test.numpy())

    ##--------------------------------------------------------------------------------
    ## Training setting
    mod_to_run = 'plmc'
    stopp_crit = 'max'
    lthresh = 5e-5 if stopp_crit == 'max' else 1e-7
    patience = 50 if stopp_crit == 'max' else 500
    train_args = {
        'gpu':False,
        'lr_max':1e-2,
        'lr_min':1e-3,
        'n_iter':3000,
        'stopp_crit':stopp_crit,
        'loss_thresh':lthresh,
        'patience':patience, 
        'last_epoch':15000,
        'print_loss':print_loss,
        'freq_print':freq_print,
    }
    ## Facultative. By default, the lr is decreased exponentially from lr_max to lr_min over n_iter iterations
    lr_max, lr_min, last_epoch = train_args['lr_max'], train_args['lr_min'], train_args['last_epoch']
    train_args['lambda_lr'] = lambda i : i/last_epoch*lr_min/lr_max + (last_epoch-i)/last_epoch if i <= last_epoch else lr_min/lr_max

    ##--------------------------------------------------------------------------------
    ## Modeling options
    ker_type = gp.kernels.RBFKernel # gp.kernels.PiecewisePolynomialKernel
    mean_type = gp.means.ZeroMean
    lik_rank = 0
    n_lat = 12
    noise_bound = 1e-4
    ker_kwargs = {}
    decomp = [[0,1,2,3]] # Do not modify (no reconstruction implemented)
    icm_kwargs = {'init_lmc_coeffs':True,
                  'n_inducing_points':None,
                  'decomp':decomp,
                  'fix_diagonal':True,
                  'diag_value':2*torch.finfo(torch.get_default_dtype()).tiny,
                  'model_type':'ICM', 
                  }
    vlmc_kwargs = {'init_lmc_coeffs':True,
                   'decomp':decomp,
                   'train_ind_ratio':1.5,
                   'seed':0, 
                   'distrib':gp.variational.CholeskyVariationalDistribution,
                   'var_strat':gp.variational.UnwhitenedVariationalStrategy,
                    }
    plmc_kwargs = {'init_lmc_coeffs':True,
                     'n_inducing_points':None,
                     'decomp':decomp,
                     'noise_thresh':1e-4,
                     'bulk':True,
                     'BDN':True,
                     'diagonal_B':True,
                     'scalar_B':True,
                     'diagonal_R':False,
                     }
    results = {}

if mod_to_run == 'icm':
    likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank, noise_constraint=gp.constraints.GreaterThan(noise_bound))
    model = MultitaskGPModel(X, Y, likelihood, n_tasks=n_tasks, n_latents=n_lat, 
                                        mean_type=mean_type, kernel_type=ker_type, ker_kwargs=ker_kwargs, **icm_kwargs)

if mod_to_run == 'vlmc':
    likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank, noise_constraint=gp.constraints.GreaterThan(noise_bound))
    model = VariationalMultitaskGPModel(X, train_y=Y, n_tasks=n_tasks,
                        mean_type=mean_type, kernel_type=ker_type, n_latents=n_lat,
                        ker_kwargs=ker_kwargs, **vlmc_kwargs)

if mod_to_run == 'plmc':
    likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_lat]), noise_constraint=gp.constraints.GreaterThan(noise_bound))
    model = ProjectedGPModel(X, Y, n_tasks, n_lat, proj_likelihood=likelihood,
                                    mean_type=mean_type, kernel_type=ker_type,
                                    ker_kwargs=ker_kwargs, **plmc_kwargs)

## Training
model, likelihood, train_metrics = train_model(model, likelihood, X, Y, train_args, compute_loo=(mod_to_run!='vlmc'), met_dict=met_dict)
if print_metrics:
    print('\n Leave-one-out errors \n')
    for key, value in train_metrics.items():
        print(key, cround(value, ndec=ndec))

## Making predictions
if make_preds:
    extra_context = {'skip_pred_var':gp.settings.skip_posterior_variances(state=isinstance(model, MultitaskGPModel))}
    # The default method for computing the variance of the ICM is memory-intensive. Switching to custom method
    test_args = {'gpu':False}
    preds, sigmas, pred_metrics = eval_model(model, likelihood, X_test, Y_test, argus=test_args, extra_context_managers=extra_context, 
                                             met_dict=met_dict)
    if print_metrics:
        print('\n Test set \n ')
        for key, value in pred_metrics.items():
            print(key, cround(value, ndec=ndec))

## Saving model and results
dico = {}
model_dico = model.save(likelihood) if mod_to_run == 'vlmc' else model.save() # The likelihood of the VLMC is dissociated from the model
dico['case'] = case
dico['X'] = X.tolist()
dico['y_means'] = means.tolist()
dico['y_maxes'] = maxes.tolist()
dico['y_labels'] = xs_keys.tolist()
dico['x_labels'] = ['Bu', 'Tf', 'Tm', 'Cb'] # Later, read from the data
dico['x_bounds'] = [[0, 62000], [373.15, 2073.15], [373.15, 600], [0, 2500]] # Later, read from the data
dico['kernel_decomp'] = decomp
dico['kernel_type'] = ker_type.__name__
dico.update(model_dico)

run_info = {}
train_dico = {}
for key,value in train_metrics.items():
    if key in met_dict:
        train_dico[key + '_loo'] = value.tolist()
    else:
        train_dico[key] = value
run_info.update(train_dico)

if make_preds:
    run_info.update(pred_metrics)
    if export_preds:
        run_info['pred_y'] = preds.tolist()
        run_info['pred_sigma'] = sigmas.tolist()

if save_torch_model:
    torch.save(model.state_dict(), exproot + 'model_{0}.pth'.format(mod_to_run))  #standard pytorch format
    dico['Y'] = Y.tolist()
    if mod_to_run == 'vlmc':
        torch.save(model.likelihood.state_dict(), exproot + 'likelihood_{0}.pth'.format(mod_to_run))

if file_format == 'pickle':
    with open(exproot + '{0}_run_info.pkl'.format(case), 'wb') as f:
        pickle.dump(run_info, f)
    with open(exproot + 'model_{0}.komi'.format(case), 'wb') as f:
        pickle.dump(dico, f)
elif file_format == 'json':
    with zipfile.ZipFile(exproot + '{0}.zip'.format(case), 'w', zipfile.ZIP_DEFLATED) as zf:
    # Écrire le dictionnaire au format JSON dans un fichier à l'intérieur de l'archive
        zf.writestr(exproot + 'gp_data_{0}.komi.json'.format(case), json.dumps(dico, indent = 3))
    with zipfile.ZipFile(exproot + '{0}.zip'.format(case), 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(exproot + 'gp_{0}_run_info.json'.format(case), json.dumps(dico, indent = 3))






