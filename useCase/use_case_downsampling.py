import os
num_threads = os.environ.get('OMP_NUM_THREADS')
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads) # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = str(num_threads) # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads) # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads) # export NUMEXPR_NUM_THREADS
import time

import numpy as np
np.random.seed(seed=12)
import pandas as pd
import torch
torch.manual_seed(12)
torch.set_default_dtype(torch.float64)
import gpytorch as gp

from komi.mogp_plmc import ProjectedGPModel
from komi.mogp_lazy import LazyLMCModel
from komi.active_sampler import ActiveSampler, get_closest_el, prod_func, max_func, sum_func
from komi.train_gp import train_model, eval_model, eval_loo
from komi.utilities import transfo_mesh, SplineKernel
##------------------------------------------------------------------------------------------------------------------------

## Data specifications
assembly_name = 'b0py26e' # this code proceeds assembly-wise. It should be generalized to a concatenation of assemblies
disc = 'FA'
ngroup = 'grp02'
mix_name = 'L_chain'
hom_name = ngroup +'_ENE_' + disc + '_' + mix_name
root = '../useCase/neutro_data/'
variables = ['root_bu','tf','tm','br']
vars_n = [var+'_n' for var in variables]
all_vars = variables + vars_n + ['bu', 'root_tf']
var_ranges = {'bu':[0, 62000], 'tf':[373.15, 2073.15], 'tm':[373.15, 600], 'br':[0, 2500]} # Later, read from the data
var_ranges['root_bu'] = np.sqrt(var_ranges['bu']).tolist()
min_bu = 75. # Restrain training and sampling to some range of Bu
max_bu = 51000.

def process_data(xs, cc, xs_keys=None, tensor_output=False):
    mask = (xs['tf'] >= xs['tm'])&(xs['bu']>min_bu)&(xs['bu']<max_bu)
    xs = xs[mask]
    cc = cc[mask]
    if xs_keys is None:
        xs_keys = xs.columns.difference(all_vars)
    train_labels = xs[xs_keys]
    xs.loc[:,'root_bu'] = np.sqrt(xs['bu'].values)
    train_x = xs[variables].copy()
    for var in variables:
        train_x[var+'_n'] = transfo_mesh(var_ranges[var], value=train_x[var])
    train_x = train_x[vars_n]
    if tensor_output:
        train_x = torch.tensor(train_x.values, dtype = torch.get_default_dtype())
        train_labels = torch.tensor(train_labels.values, dtype = torch.get_default_dtype())
    return train_x, train_labels, cc 

##---------------------------------------------------------------------------------------------------------------------------------
## Training data
norm_func = torch.std
init_tag = 'sobol_PIJ_256'
init_data = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + init_tag + '_xs.csv', index_col=0)
init_cc = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + init_tag + '_cc.csv', index_col=0)
train_x, train_labels, init_cc = process_data(init_data, init_cc)
print('Initial number of points :', len(train_x))
mask = (init_data['tf'] >= init_data['tm'])&(init_data['bu']>min_bu)&(init_data['bu']<max_bu)
init_data = init_data[mask] # for the final export

if ngroup=='grp20' or mix_name=='L_chain':
    filt = (np.abs(train_labels).mean(axis=0) > 1e-5) # Filter out the labels with low mean values
    train_labels = train_labels.loc[:,filt]
xs_keys = train_labels.columns
cc_keys = init_cc.columns.difference(all_vars)
train_x = torch.tensor(train_x.values, dtype = torch.get_default_dtype())
train_labels = torch.tensor(train_labels.values, dtype = torch.get_default_dtype())
means, devs = train_labels.mean(dim=0), norm_func(train_labels, dim=0)
train_labels = (train_labels - means) / devs

## Test data
external_tests = True
if external_tests:
    test_tag = 'LHS_PIJ_512'
    test_data = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + test_tag + '_xs.csv', index_col=0)
    test_cc = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + test_tag + '_cc.csv', index_col=0)
    test_x, test_labels, test_cc = process_data(test_data, test_cc, xs_keys=xs_keys, tensor_output=True)
    test_labels = (test_labels - means) / devs

##------------------------------------------------------------------------------------------------------------------------

## Sampling setings
strategy = 'Tdownsampling'
# task-level scores. 'Ldownsampling' for latent-level scores
mod_to_run = 'plmc'
retrain = False
aggr_func = prod_func
renormalize = True
batch_size = 1
final_set_size = 10
stopp_crit = 'RMSE'
stopp_tol = 2e-1 # If the deterioration of the stopp_crit metric is above this threshold, the sampling is stopped
n_tests = 10
verbose = True

n_steps = len(train_x) - final_set_size
points_iter = range(0, n_steps, batch_size)
freq_test = len(points_iter) // n_tests if (n_tests > 0 and external_tests) else len(points_iter) + 1
study_tag_base = '{0}adapt_{1}'.format(strategy, mod_to_run)
if stopp_tol is None:
    study_tag_base += '_{0}pts'.format(final_set_size)
else:
    study_tag_base += '_tol{0}'.format(stopp_tol)
if not retrain:
    study_tag_base += '_noretrain'
print('Name of the current experiment :', study_tag_base)

##------------------------------------------------------------------------------------------------------------------------
## Training settings
train_stopp_crit = 'exp'
sched = 'lin'
lthreshes = {'max':1e-5, 'mean':1e-7, 'exp':1e-9}
patiences = {'max':50, 'mean':500, 'exp':50}
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
train_args = {
    'gpu':False,
    'lr_max':1e-2,
    'lr_min':2e-3,
    'n_iter':50000,
    'stopp_crit':train_stopp_crit,
    'loss_thresh':lthreshes[train_stopp_crit],
    'patience':patiences[train_stopp_crit], 
    'print_loss':print_loss,
    'freq_print':freq_print,
}
## Facultative. By default, the lr is decreased exponentially from lr_max to lr_min over n_iter iterations
## Also, incompatible with parallel training of SOGPs (lambda functions are not pickable)
if sched == 'lin' and mod_to_run != 'sogp':
    last_epoch = train_args['n_iter'] - 1
    lr_max, lr_min = train_args['lr_max'], train_args['lr_min']
    train_args['lambda_lr'] = lambda i : i/last_epoch*lr_min/lr_max + (last_epoch-i)/last_epoch if i <= last_epoch else lr_min/lr_max

##------------------------------------------------------------------------------------------------------------------------
    
## GP settings
ker_type = gp.kernels.MaternKernel
n_lat = 12
noise_bound = 1e-4
plmc_kwargs = {'init_lmc_coeffs':True,
                    'n_inducing_points':None,
                    'decomp':None,
                    'bulk':True,
                    'BDN':True,
                    'diagonal_B':True,
                    'scalar_B':True,
                    'diagonal_R':False,
                    'ortho_param':'matrix_exp',
                    'noise_thresh':noise_bound,
                    'jitter_val':1e-8,  
                    }
lazy_kwargs = {'noise_val':1e-7, 'store_full_y':True,}

if mod_to_run == 'plmc':
    model = ProjectedGPModel(train_x, train_labels, n_lat, kernel_type=ker_type, **plmc_kwargs)
elif mod_to_run == 'lazy_lmc':
    model = LazyLMCModel(train_x, train_labels, n_lat, **lazy_kwargs)

met_dict = {
    'R2': lambda rwm : (1 - torch.mean(rwm['errs2'], dim=0) / torch.var(rwm['y_test'], dim=0)).mean(),
    'RMSE': lambda rwm : torch.sqrt(rwm['errs2'].mean()),
    'mean_err_abs': lambda rwm : rwm['errs'].mean(),
    'max_err_abs': lambda rwm : rwm['errs'].max(),
}
##------------------------------------------------------------------------------------------------------------------------
## Iteration

sampler = ActiveSampler(model, strategy, aggr_func, current_data=train_labels, current_X=train_x)

progress = {}
start = time.time()
tstep = start                    
first_run = True
optimizer = None
for k in points_iter:
    print('\n Current iter :', k)

    # Data update
    sampler.modify_train_set(new_X=None, new_Y=None, normalize=renormalize, norm_func=norm_func)

    # Model training and point query
    if (first_run or retrain) and mod_to_run != 'lazy_lmc':
        sampler.model, train_stats, optimizer = train_model(sampler.model, train_x, train_labels, train_args, compute_loo=True,
                                                    optimizer=optimizer, met_dict=met_dict, return_optim=True)
    else:
        train_stats = eval_loo(model, train_labels, met_dict=met_dict) # only to track the sampling process

    new_points, best_score = sampler.find_next_points(n_samples=batch_size)
    new_time = time.time()
    tstep = new_time
    progress[k] = {'total_time':new_time - start, 'time':new_time - tstep, 'score':best_score, 'av_cond':model.kernel_cond().cpu().numpy().mean()}
    progress[k].update(train_stats)
    stopp_crit_val = progress[k][stopp_crit]
    if verbose:
        print(progress[k])

    make_test = (k % freq_test == 0) or (n_tests > 0 and k == list(points_iter)[-1])
    if make_test:
        test_args = {'gpu':False}
        __, __, test_stats = eval_model(model, test_x, test_labels, test_args, met_dict=met_dict)
        print('\n Test results : \n')
        for key, value in test_stats.items():
            print(key, value)
        progress[k].update(test_stats)

    if first_run:
        first_stopp_crit_val = stopp_crit_val
        train_args['lr_max'] = train_args['lr_min'] # After initial training, the learning rate is kept minimal
        train_args['n_iter'] = 10000 # For a shorter sampling process ; this bound is rarely attained anyway. Can also evolve during sampling
        first_run = False     
    
    if 1 - first_stopp_crit_val / stopp_crit_val > stopp_tol:
        break
    i = k + batch_size - 1

    ##------------------------------------------------------------------------------------------------------------------------
        
    ## Results storage
    ## Currently, the results are stored in full at each iteration to avoid data loss in case of interruption.
    output_tag = study_tag_base
    output_path = 'results/sampled_sets/'
    output_path_xs = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_xs.csv'
    output_path_cc = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_cc.csv'
    visited_mask = np.array([point not in sampler.visited_points for point in np.arange(len(train_x))]).astype(bool)
    df_xs = init_data.loc[visited_mask,:]
    df_cc = init_cc.loc[visited_mask,:]
    df_xs.to_csv(output_path_xs)
    df_cc.to_csv(output_path_cc)

    complement_data = init_data.iloc[sampler.visited_points,:]
    complement_cc = init_cc.iloc[sampler.visited_points,:]
    output_tag_comp = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_complement_xs.csv'
    output_cctag_comp = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_complement_cc.csv'
    complement_data.to_csv(output_tag_comp)
    complement_cc.to_csv(output_cctag_comp)

    df_progress = pd.DataFrame.from_dict(progress, orient='index')
    output_path_progress = 'results/sampling_runs/' + assembly_name + '_' + hom_name + '_' + output_tag + '_progress.csv'
    df_progress.to_csv(output_path_progress)
    ##---------------------------------------------------------------------------------------------------------------------
