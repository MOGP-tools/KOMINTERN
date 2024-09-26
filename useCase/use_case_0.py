import pickle
import json
import zipfile
import resource

import os
num_threads = os.environ.get('OMP_NUM_THREADS')
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads) # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = str(num_threads) # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads) # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads) # export NUMEXPR_NUM_THREADS

import numpy as np
np.random.seed(seed=12)
import pandas as pd

import torch
torch.manual_seed(12)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float64)
import gpytorch as gp

from komi.utilities import transfo_mesh, SplineKernel, LinearMean, PolynomialMean, max_norm_func, compute_macro_errs
from komi.base_gp import ExactGPModel
from komi.mogp_lazy import LazyLMCModel
from komi.mogp_var import VariationalMultitaskGPModel
from komi.mogp_plmc import ProjectedGPModel, ProjectedLMCmll
from komi.mogp_icm import MultitaskGPModel
from komi.train_gp import train_model, eval_model, eval_loo, train_parallel, eval_parallel

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
    # 'mean_err_rel': lambda rwm : (rwm['errs'] / torch.std(rwm['y_test'], dim=0)).mean(),
    # 'max_err_rel': lambda rwm : (rwm['errs'] / torch.std(rwm['y_test'], dim=0)).max(),
    'mean_err_quant05': lambda rwm : torch.quantile(rwm['errs'], 0.05),
    'mean_err_quant95': lambda rwm : torch.quantile(rwm['errs'], 0.95),
    'mean_err_quant99': lambda rwm : torch.quantile(rwm['errs'], 0.99),
    'mean_err_macro': lambda rwm : compute_macro_errs(rwm['errs'], rwm['concs'], rwm['keys']).mean(),
    'max_err_macro': lambda rwm : compute_macro_errs(rwm['errs'], rwm['concs'], rwm['keys']).max(),
}

print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
ndec = 2
compute_loo = True
make_preds = True
export_preds = False
export_data_to_np = False
save_torch_model = True
file_format = 'json' # 'json' or 'pickle'

if __name__ == "__main__":

    ## Data preparation
    root = '../useCase/neutro_data/'

    variables = ['root_bu', 'tf', 'tm', 'br']
    vars_n = [var + '_n' for var in variables] #normalized variables
    all_vars = variables + vars_n + ['bu']
    var_ranges = {'bu':[0, 62000], 'tf':[373.15, 2073.15], 'tm':[373.15, 600], 'br':[0, 2500]} # Later, read from the data
    var_ranges['root_bu'] = np.sqrt(var_ranges['bu']).tolist()

    train_df = pd.read_csv(root + 'b0py26e_grp02_ENE_FA_L_chain_downsampled_xs.csv', index_col=0)
    test_df = pd.read_csv(root + 'b0py26e_grp02_ENE_FA_L_chain_LHS_PIJ_512_xs.csv', index_col=0)
    test_cc = pd.read_csv(root + 'b0py26e_grp02_ENE_FA_L_chain_LHS_PIJ_512_cc.csv', index_col=0)

    filt = (np.abs(train_df).mean(axis=0) > 1e-5) #elimintation of almost-zero HXS
    train_df = train_df.loc[:, filt]
    min_bu, max_bu = 75., 51000.
    mask = (train_df['bu']>min_bu) & (train_df['bu']<max_bu) & (train_df['tf']>train_df['tm']) #elimination of points with bu outside the range
    train_df = train_df[mask]
    mask = (test_df['bu']>min_bu) & (test_df['bu']<max_bu) & (test_df['tf']>test_df['tm'])
    test_df = test_df[mask]
    xs_keys = train_df.columns.difference(all_vars) #just HXS, no variables
 
    norm_func = torch.std
    # norm_func = max_norm_func
    Y = torch.as_tensor(train_df[xs_keys].values)
    Y_test = torch.as_tensor(test_df[xs_keys].values)
    means, devs = Y.mean(dim=0), norm_func(Y, dim=0)
    Y = (Y - means) / devs
    Y_test = (Y_test - means) / devs

    train_df['root_bu'] = np.sqrt(train_df['bu'].values)
    test_df['root_bu'] = np.sqrt(test_df['bu'].values)
    train_x = train_df[variables].copy()
    test_x = test_df[variables].copy()
    for var in variables:
        train_x[var + '_n'] = transfo_mesh(var_ranges[var], value=train_x[var])
        test_x[var + '_n'] = transfo_mesh(var_ranges[var], value=test_x[var])
    X = torch.as_tensor(train_x[vars_n].values)
    X_test = torch.as_tensor(test_x[vars_n].values)

    # X, X_test, Y, Y_test = X[::2], X_test[::2], Y[::2, :15], Y_test[::2, :15]
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
    mod_to_run = 'vlmc' # 'icm', 'plmc', 'vlmc', 'lazy_lmc', 'sogp'
    stopp_crit = 'exp'
    sched = 'lin'
    lthreshes = {'max':1e-5, 'mean':1e-7, 'exp':1e-10}
    patiences = {'max':50, 'mean':500, 'exp':50}
    train_args = {
        'gpu':False,
        'lr_max':1e-2,
        'lr_min':2e-3,
        'n_iter':20000,
        'stopp_crit':stopp_crit,
        'loss_thresh':lthreshes[stopp_crit],
        'patience':patiences[stopp_crit], 
        'print_loss':print_loss,
        'freq_print':freq_print,
    }
    ## Facultative. By default, the lr is decreased exponentially from lr_max to lr_min over n_iter iterations
    ## Also, incompatible with parallel training of SOGPs (lambda functions are not pickable)
    if sched == 'lin' and mod_to_run != 'sogp':
        last_epoch = train_args['n_iter'] - 1
        lr_max, lr_min = train_args['lr_max'], train_args['lr_min']
        train_args['lambda_lr'] = lambda i : i/last_epoch*lr_min/lr_max + (last_epoch-i)/last_epoch if i <= last_epoch else lr_min/lr_max

    ##--------------------------------------------------------------------------------
    ## Modeling options
    ker_type = gp.kernels.RBFKernel # gp.kernels.PiecewisePolynomialKernel
    mean_type = gp.means.ZeroMean # Do not change except good reason !
    lik_rank = 0
    n_lat = 12
    noise_bound = 1e-4
    decomp = [[0,1,2,3]] # Do not modify (no reconstruction implemented)
    icm_kwargs = {'init_lmc_coeffs':True,
                  'n_inducing_points':None,
                  'decomp':decomp,
                  'fix_diagonal':True,
                  'diag_value':16*torch.finfo(torch.get_default_dtype()).tiny,
                  'model_type':'ICM',
                  'noise_thresh':noise_bound, 
                  }
    vlmc_kwargs = {'init_lmc_coeffs':True,
                   'decomp':decomp,
                   'train_ind_ratio':1.5,
                   'seed':0, 
                   'distrib':gp.variational.DeltaVariationalDistribution, #gp.variational.CholeskyVariationalDistribution,
                   'var_strat':gp.variational.UnwhitenedVariationalStrategy,
                   'noise_thresh':noise_bound,                
                   'jitter_val':1e-8,  
                    }
    plmc_kwargs = {'init_lmc_coeffs':True,
                     'n_inducing_points':None,
                     'decomp':decomp,
                     'bulk':False,
                     'BDN':True,
                     'diagonal_B':True,
                     'scalar_B':True,
                     'diagonal_R':False,
                     'ortho_param':'matrix_exp',
                     'noise_thresh':noise_bound,
                     'noise_init':noise_bound,
                     'jitter_val':1e-8,  
                     }
    lazy_kwargs = {'noise_val':1e-7,
                   'store_full_y':True,
                    }
    sogp_args = {'n_inducing_points':None,
                 'noise_thresh':1e-6,
                 'decomp':decomp,
                 'kernel_type':ker_type,
                 'mean_type':mean_type,
                 'train_x':X,
                }           

    if mod_to_run == 'icm':
        model = MultitaskGPModel(X, Y, n_latents=n_lat, mean_type=mean_type, kernel_type=ker_type, **icm_kwargs)

    if mod_to_run == 'vlmc':
        model = VariationalMultitaskGPModel(X, train_y=Y, n_tasks=n_tasks, mean_type=mean_type, kernel_type=ker_type, 
                                            n_latents=n_lat, **vlmc_kwargs)
    if mod_to_run == 'plmc':
        model = ProjectedGPModel(X, Y, n_lat, kernel_type=ker_type, **plmc_kwargs)

    if mod_to_run == 'lazy_lmc':
        X, X_test = (X+1)/2, (X_test+1)/2 # The SplineKernel is only defined on [0;1]^d, while our data is normalized on [-1;1]^d
        model = LazyLMCModel(X, Y, n_lat, **lazy_kwargs)

    if mod_to_run == 'sogp':
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1])) # Increase the limit of open files. Necessary because of parallelism
        max_workers = 2
        mod_list = [None]*n_tasks
        for i in range(n_tasks):
            args_spec = sogp_args.copy()
            args_spec['train_y'] = Y[:, i]#.unsqueeze(-1)
            mod_list[i] = (ExactGPModel, args_spec, {})

    ## Training
    if mod_to_run == 'lazy_lmc': # no need to train this model
        train_metrics = eval_loo(model, Y, met_dict=met_dict)
    elif mod_to_run == 'sogp':
        # model, train_metrics = train_model(mod_list[0], X, Y[:,0], train_args, compute_loo=(mod_to_run!='vlmc'), met_dict=met_dict)
        mod_list, train_metrics = train_parallel(mod_list, X, Y, train_args, max_workers=max_workers, compute_loo=True, met_dict=met_dict)
    else:
        model, train_metrics = train_model(model, X, Y, train_args, compute_loo=(mod_to_run!='vlmc'), met_dict=met_dict)
    if print_metrics:
        print('\n Training statistics \n')
        for key, value in train_metrics.items():
            print(key, cround(value, ndec=ndec))

    ## Making predictions
    if make_preds:
        extra_context = {'skip_pred_var':gp.settings.skip_posterior_variances(state=isinstance(model, MultitaskGPModel))}
        # The default method for computing the variance of the ICM is memory-intensive. Switching to custom method
        test_args = {'gpu':False}
        if mod_to_run == 'sogp':
            preds, sigmas, pred_metrics = eval_model(model, X_test, Y_test, argus=test_args, max_workers=max_workers, 
                                                     extra_context_managers=extra_context, met_dict=met_dict)
            preds, sigmas, pred_metrics = eval_parallel(mod_list, X_test, Y_test, argus=test_args,
                                                         extra_context_managers=extra_context, met_dict=met_dict)
        else:
            preds, sigmas, pred_metrics = eval_model(model, X_test, Y_test, argus=test_args, max_workers=max_workers, 
                                                     extra_context_managers=extra_context, met_dict=met_dict)
        if print_metrics:
            print('\n Prediction metrics \n ')
            for key, value in pred_metrics.items():
                print(key, cround(value, ndec=ndec))

    ## Saving model and results
    dico = {}
    if mod_to_run != 'sogp':
        model_dico = model.save()
        dico['case'] = case
        dico['X'] = X.tolist()
        dico['y_means'] = means.tolist()
        dico['y_devs'] = devs.tolist()
        dico['y_labels'] = xs_keys.tolist()
        dico['x_labels'] = ['Bu', 'Tf', 'Tm', 'Cb'] # Later, read from the data
        dico['x_bounds'] = var_ranges
        dico['kernel_decomp'] = decomp
        dico['kernel_type'] = ker_type.__name__
        dico.update(model_dico)

        if save_torch_model:
            torch.save(model.state_dict(), exproot + 'model_{0}.pth'.format(mod_to_run))  #standard pytorch format
            dico['Y'] = Y.tolist()

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
        for stat, value in run_info.items():
            if isinstance(value, np.ndarray):
                run_info[stat] = value.tolist()
        if export_preds:
            run_info['pred_y'] = preds.tolist()
            run_info['pred_sigma'] = sigmas.tolist()

    if file_format == 'pickle':
        with open(exproot + '{0}_run_info.pkl'.format(case), 'wb') as f:
            pickle.dump(run_info, f)
        if dico:
            with open(exproot + 'model_{0}.komi'.format(case), 'wb') as f:
                pickle.dump(dico, f)
    elif file_format == 'json':
        with zipfile.ZipFile(exproot + '{0}.zip'.format(case), 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(exproot + 'gp_{0}_run_info.json'.format(case), json.dumps(run_info, indent = 3))
        if dico:
            with zipfile.ZipFile(exproot + '{0}.zip'.format(case), 'w', zipfile.ZIP_DEFLATED) as zf:
            # Écrire le dictionnaire au format JSON dans un fichier à l'interieur de l'archive
                zf.writestr(exproot + 'gp_data_{0}.komi.json'.format(case), json.dumps(dico, indent = 3))






