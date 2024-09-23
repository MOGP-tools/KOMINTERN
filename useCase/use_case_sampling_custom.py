import os
num_threads = os.environ.get('OMP_NUM_THREADS')
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads) # export OPENBLAS_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = str(num_threads) # export MKL_NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads) # export VECLIB_MAXIMUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads) # export NUMEXPR_NUM_THREADS
import subprocess
import concurrent.futures
import time
import sys

import numpy as np
np.random.seed(seed=12)
import pandas as pd
import h5py
import torch
torch.manual_seed(12)
torch.set_default_dtype(torch.float64)
import gpytorch as gp

from komi.mogp_plmc import ProjectedGPModel
from komi.active_sampler import ActiveSampler, get_closest_el, prod_func, max_func, sum_func
from komi.train_gp import train_model, eval_model, eval_loo
from komi.utilities import transfo_mesh

##------------------------------------------------------------------------------------------------------------------------
## Apollo setup
launch_mode = 'reprise' # 'RAD' (reprise and depletion) or 'reprise'
a3_build = 'your-apollo_build'
a3_dataprefix = 'your-data-prefix'
scheme_lattice = 'sigar-folder-where-lattice-scripts-are'
scheme_data = 'sigar-folder-where-data-class-is'
pythonpath = os.environ.get('PYTHONPATH', '').split(os.pathsep)
new_pythonpath = ":".join([]) # list all the paths you want to add to the PYTHONPATH
new_path = ":".join([]) # list all the paths you want to add to the PATH
ld_library_path = ":".join([]) # list all the paths you want to add to the PATH

env_specs = {"A3_BUILD":a3_build, "A3_DATAPREFIX":a3_dataprefix, "SCHEME_LATTICE":scheme_lattice,
             "SCHEME_DATA":scheme_data, "PYTHONPATH":new_pythonpath, "PATH":new_path, "LD_LIBRARY_PATH":ld_library_path,
             "OMP_NUM_THREADS":str(num_threads)}

def run_apollo(args):
    try:
        completed = subprocess.run(['xargs', sys.executable, 'sigar_launcher.py'], input=args, capture_output=True, text=True,
                                    env=env_specs, check=True)
        return completed.stdout
    except Exception as e:
        return e
##------------------------------------------------------------------------------------------------------------------------

## Data specifications
assembly_name = 'b24gd31e' # this code proceeds assembly-wise. It should be generalized to a concatenation of assemblies
disc = 'FA'
ngroup = 'grp02'
mix_name = 'L_chain'
hom_name = ngroup +'_ENE_' + disc + '_' + mix_name
root = os.path.join('/home/catB/ot266455/SIGAR/SCHEME/LATTICE', assembly_name, 'REPRISE', 'export_xs', hom_name,'')
xs_path = assembly_name + '/REPRISE/export_xs/'+ hom_name + '/'
variables = ['root_bu','tf','tm','br']
vars_n = [var+'_n' for var in variables]
all_vars = variables + vars_n + ['bu', 'root_tf']
var_ranges = {'bu':[0, 62000], 'tf':[373.15, 2073.15], 'tm':[373.15, 600], 'br':[0, 2500]} # Later, read from the data
var_ranges['root_bu'] = np.sqrt(var_ranges['bu']).tolist()
min_bu = 75. # Restrain training and sampling to some range of Bu
max_bu = 51000.

def process_data(xs, cc):
    mask = (xs['tf'] >= xs['tm'])&(xs['bu']>min_bu)&(xs['bu']<max_bu)
    xs = xs[mask]
    cc = cc[mask]
    xs_keys = xs.columns.difference(all_vars)
    train_labels = xs[xs_keys]
    xs['root_bu'] = np.sqrt(xs['bu'].values)
    train_x = xs[variables].copy()
    for var in variables:
        train_x[var+'_n'] = transfo_mesh(var_ranges[var], value=train_x[var])
    train_x = train_x[vars_n]
    return train_x, train_labels, cc 

##---------------------------------------------------------------------------------------------------------------------------------
## Training data
strategy = 'Tdownsampling'

init_tag = 'sobol_PIJ_256' if strategy in ['Ldownsampling', 'Tdownsampling'] else 'sobol_PIJ_64'
init_data = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + init_tag + '_unstruct_0_xs.csv', index_col=0)
init_cc = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + init_tag + '_unstruct_0_cc.csv', index_col=0)
train_x, train_labels, init_cc = process_data(init_data, init_cc)

if strategy not in ['Ldownsampling', 'Tdownsampling']:
    red_factor = 1 # In case one wants to reduce the initial number of points
    init_data, init_cc = init_data.iloc[::red_factor, :], init_cc.iloc[::red_factor, :]
print('Initial number of points :', len(init_data))

if ngroup=='grp20' or mix_name=='L_chain':
    filt = (np.abs(train_labels).mean(axis=0) > 1e-5) # Filter out the labels with low mean values
    train_labels = train_labels.loc[:,filt]
xs_keys = train_labels.columns
cc_keys = init_cc.columns.difference(all_vars)

## Test data
external_tests = True
if external_tests:
    test_tag = 'LHS_PIJ_512'
    test_data = pd.read_csv(root + assembly_name + '_' + hom_name + '_' + test_tag + '_unstruct_0_xs.csv', index_col=0)
    test_cc = pd.read_csv(root + 'something-with-test_tag' + '_xs.csv', index_col=0)
    test_x, test_labels, test_cc = process_data(test_data, test_cc)
    test_labels = test_labels[xs_keys]

## Candidate data
dummy_mode = True
# If True, the script will not run the lattice code : instead it will draw samples from a preobtained dataset
if strategy in ['Ldownsampling', 'Tdownsampling']:
    pass # no candidate data in this case
elif dummy_mode:
    candidate_tag = 'some-large-dataset'
    candidate_df = pd.read_csv(root + 'something-with-candidate_tag' + '_xs.csv', index_col=0)
    candidate_cc = pd.read_csv(root + 'something-with-candidate_tag' + '_cc.csv', index_col=0)
    candidate_x, candidate_labels, candidate_cc = process_data(candidate_df, candidate_cc)
else: # No candidate data, but an evolution archive is required
    if launch_mode=='RAD':
        archive_path = 'some-archive-path' + init_tag + '.hdf'
    elif launch_mode=='reprise':
        archive_path = 'path-of-some-large-archive' + '.hdf'
    file = h5py.File(archive_path,'r')
    bu_depl = file['parameters']['values']['PARAM_1'][:]

##------------------------------------------------------------------------------------------------------------------------

## Sampling setings
retrain = False
norm_func = torch.std
aggr_func = prod_func
renormalize = True
batch_size = 1
final_set_size = 400
n_steps = final_set_size - len(train_x) if strategy not in ['Tdownsampling', 'Ldownsampling'] else len(train_x) - final_set_size
first_point = len(train_x) if strategy not in ['Tdownsampling', 'Ldownsampling'] else 0
points_iter = range(first_point, first_point + n_steps, batch_size)
n_tests = 10
freq_test = len(points_iter) // n_tests if (n_tests > 0 and external_tests) else len(points_iter) + 1
study_tag_base = '{0}adapt_{1}p'.format(strategy, final_set_size)
if not retrain:
    study_tag_base += '_noretrain'
print('Name of the current experiment :', study_tag_base)
arg_ass = '-assembly ' + assembly_name
arg_tag = '-study_tag ' + study_tag_base

##------------------------------------------------------------------------------------------------------------------------
## Training settings
mod_to_run = 'plmc'
stopp_crit = 'exp'
sched = 'lin'
lthreshes = {'max':1e-5, 'mean':1e-7, 'exp':1e-9}
patiences = {'max':50, 'mean':500, 'exp':50}
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
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
                    'noise_init':noise_bound,
                    'jitter_val':1e-8,  
                    }
model = ProjectedGPModel(train_x, train_labels, n_lat, kernel_type=ker_type, **plmc_kwargs)

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
##------------------------------------------------------------------------------------------------------------------------
## Iteration

progress = {}
start = time.time()
tstep = start                    
cc_data_res = {}
xs_data_res = {}
sampler = ActiveSampler(model, strategy, aggr_func, current_data=train_labels, current_X=train_x)
if strategy not in ['Tdownsampling', 'Ldownsampling']:
    sampler.gen_candidate_set(1000, dim=len(variables), algo='sobol', seed=12, return_set=False)
first_run = True
for k in points_iter:
    print('\n Current iter :', k)

    # Model training and point query
    if first_run or retrain:
        model, train_stats, optimizer = train_model(model, train_x, train_labels, train_args, 
                                                    optimizer=optimizer, met_dict=met_dict, return_optim=True)
    else:
        train_stats = eval_loo(model, train_labels, met_dict=met_dict) # only to track the sampling process
    new_points, best_score = sampler.find_next_points()
    new_time = time.time()
    tstep = new_time
    k = first_point
    progress[k] = {'total_time':new_time - start, 'time':new_time - tstep, 'score':best_score}
    progress[k].update(train_stats)

    make_test = (k % freq_test == 0) or (n_tests > 0 and k == list(points_iter)[-1])
    if make_test:
        test_stats = eval_model(model, test_x, test_labels, met_dict=met_dict)
        progress[k].update(test_stats)
        if strategy not in ['Tdownsampling', 'Ldownsampling']:
            score_corr = sampler.compute_corr(test_x, test_labels) # compute correlation between uncertainty score and errors
            progress[k]['score_corr'] = score_corr

    if first_run:
        train_args['lr_max'] = train_args['lr_min'] # After initial training, the learning rate is kept minimal
        train_args['n_iter'] = 10000 # For a shorter sampling process ; this bound is rarely attained anyway. Can also evolve during sampling
        first_run = False
          
    # New data generation
    if not dummy_mode or strategy in ['Tdownsampling', 'Ldownsampling']:
        for j, var in enumerate(variables):
            new_points[:, j] = transfo_mesh(var_ranges[var], value=new_points[:, j], reverse=True)

        last_stage = False
        is_finished = False
        while not is_finished:
            args_list = []
            for i, point in enumerate(new_points):
                point = point.copy()
                bu = point[0]**2 # HXS's are parametrized in root(Bu)
                point[0] = bu          
                closest_bu = get_closest_el(bu_depl, bu)
                arg_ntag = '-subset_tag ' + str(k+i)

                if launch_mode=='reprise':
                    point[0] = closest_bu
                    # In this mode, the point queried by the sampling algorithm is modified so that its Bu value matches the closest value in a large archive
                    arg_ctype = '-ctype ' + 'REPRISE'
                    arg_path = '-archive_path ' + archive_path
                    arg_point = '-point ' + ' '.join(str(val) for val in point)
                                    
                elif launch_mode=='RAD' and not last_stage:
                    # In this mode, an evolution calculation is performed between the closest Bu value in the archive and the queried one
                    closest_bu = get_closest_el(bu_depl, bu)
                    bu_it = [closest_bu.item(), bu.item()]
                    arg_ctype = '-ctype ' + 'REPRISE_AND_DEPLETION'
                    arg_path = '-archive_path ' + archive_path
                    arg_point = '-point ' + ' '.join(str(val) for val in bu_it)

                elif launch_mode=='RAD' and last_stage:
                    arg_ctype = '-ctype ' + 'REPRISE'
                    rad_path = 'path-of-the-previously-generated-archive'.format(k+i) + '.hdf'
                    arg_path = '-archive_path ' + rad_path
                    arg_point = '-point ' + ' '.join(str(val) for val in point)
                else:
                    raise ValueError('Launch mode not recognized')

                args = ' '.join([arg_ass, arg_path, arg_ctype, arg_point, arg_tag, arg_ntag])
                args_list.append(args)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                try:
                    results = list(executor.map(run_apollo, args_list))
                except Exception as e:
                    print("An error occurred:", e)

            if launch_mode == 'reprise' or last_stage :
                is_finished = True
            else:
                last_stage = True

        ## Processing of the generated data
        new_xs_items, new_cc_items = [], []
        for i in range(k, k + batch_size):
            study_tag = study_tag_base + '_' + str(i)
            xs_data = pd.read_csv(xs_path + study_tag_base + '/' + assembly_name + '_' + hom_name + '_' + study_tag + '_xs.csv', index_col=0) # modify as needed
            cc_data = pd.read_csv(xs_path + study_tag_base + '/' + assembly_name + '_' + hom_name + '_' + study_tag + '_cc.csv', index_col=0) # modify as needed
            xs_data_res.update({i : xs_data.to_dict(orient='index')['0_0']}) #  The 0_0 string is due to the export format of the HXS csv's
            cc_data_res.update({i : cc_data.to_dict(orient='index')['0_0']})
            new_xs_items.append(xs_data)
            new_cc_items.append(cc_data)
        new_xs_data = pd.concat(new_xs_items, axis=0)
        new_cc_data = pd.concat(new_cc_items, axis=0)
        new_xs_data['root_bu'] = np.sqrt(new_xs_data['bu'].values)
        new_train_x, new_train_labels = new_xs_data[variables].copy(), new_xs_data[xs_keys].copy()
        for var in variables:
            new_train_x[var + '_n'] = transfo_mesh(var_ranges[var], value=new_train_x[var])
        new_train_x = new_train_x[vars_n]

    else:
        if strategy in ['Tdownsampling', 'Ldownsampling']:
            i = k + batch_size - 1
            new_train_labels, new_train_x = None, None
        else:
            new_train_x = candidate_x.iloc[new_points,:]
            new_train_labels = candidate_labels.iloc[new_points,:]
            part_xs = candidate_df.iloc[new_points,:]
            part_cc = candidate_cc.iloc[new_points,:]
            for i in range(k, k + batch_size):
                if batch_size==1:
                    xs_data_res.update({i : part_xs.to_dict()})
                    cc_data_res.update({i : part_cc.to_dict()})
                else:
                    xs_data_res.update({i : part_xs.to_dict(orient='index')['0_0']})
                    cc_data_res.update({i : part_cc.to_dict(orient='index')['0_0']})

    ## Model data update
    sampler.modify_train_set(new_X=new_train_x, new_Y=new_train_labels, normalize=renormalize, norm_func=norm_func)
    ##------------------------------------------------------------------------------------------------------------------------
        
    ## Results storage
    ## Currently, the results are stored in full at each iteration to avoid data loss in case of interruption.
    output_tag = study_tag_base
    output_path = '/home/catB/ot266455/XS_modeling/cell_calculation/results_gp/'
    output_path_xs = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_xs.csv'
    output_path_cc = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_cc.csv'
    if strategy in ['Tdownsampling', 'Ldownsampling']:
        visited_mask = np.array([point not in sampler.visited_points for point in np.arange(len(train_x))]).astype(bool)
        df_xs = init_data.loc[visited_mask,:]
        df_cc = init_cc.loc[visited_mask,:]
        complement_data = init_data.iloc[sampler.visited_points,:]
        complement_cc = init_cc.iloc[sampler.visited_points,:]
        output_tag_comp = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_complement_xs.csv'
        output_cctag_comp = output_path + assembly_name + '_' + hom_name + '_' + output_tag + '_complement_cc.csv'
        complement_data.to_csv(output_tag_comp)
        complement_cc.to_csv(output_cctag_comp)
    else:
        df_xs = pd.DataFrame.from_dict(xs_data_res, orient='index')
        df_cc = pd.DataFrame.from_dict(cc_data_res, orient='index')
    df_xs.to_csv(output_path_xs)
    df_cc.to_csv(output_path_cc)
    df_progress = pd.DataFrame.from_dict(progress, orient='index')
    output_path_progress = 'results/sampling_runs/' + assembly_name + '_' + hom_name + '_' + output_tag + '_progress.csv'
    df_progress.to_csv(output_path_progress)
    ##---------------------------------------------------------------------------------------------------------------------
