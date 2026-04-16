
import time

import numpy as np
np.random.seed(seed=12)
import pandas as pd
import torch
torch.manual_seed(12)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float32)
from torch.utils.data import TensorDataset, DataLoader
import gpytorch as gp
import wandb

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from base_gp import ExactGPModel
from mogp_icm import MultitaskGPModel
from mogp_var import VariationalMultitaskGPModel
from mogp_plmc import ProjectedGPModel

from utilities import CorrectReduceLROnPlateau

##------------------------------------------------------------------
## Generic setup 

print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
# For experiments where all models have inducing points, this variable is overriden
gpu = True
export_results = True
appendix = '' # to further customize experiment name
experiments = ['neutro', 'ship', 'sarcos', 'tidal']
models_to_run = ['PLMC', 'PLMC_fast', 'oilmm', 'var', 'ICM']
models_to_run = ['IGP']
v_test_2 = 'void'
##----------------------------------------------

## >>> Reproducing the experiments of the paper : just chose the experiment of interest here, then run the script <<<
experiment = experiments[-1]

## There is only one subtelty : for models 'var' and 'ICM', two options have been tested for the likelihood rank, 0 and n_tasks.
## Running these two options by setting v_test_2=lik_rank would result in duplicated results for all other models.
## Therefore, if you want to perform this test on 'var' and 'ICM', you can uncomment the following lines.

# models_to_run = ['var','ICM']
# v_test_2 = 'lik_rank'
##----------------------------------------------

def compute_metrics(y_test, y_pred, std_pred, loss, Sigma_guess, n_iter, train_time, pred_time, print_metrics=True):
    delta = y_test - y_pred
    errs_abs = torch.abs(delta).squeeze()
    alpha_CI = torch.mean((errs_abs < 2 * std_pred).float())
    err2 = errs_abs ** 2
    R2_list = 1 - torch.mean(err2, dim=0) / torch.var(y_test, dim=0)
    PVA_list = torch.log(torch.mean(err2 / std_pred ** 2, dim=0))
    noise_full = Sigma_guess.diag().mean() # mean of the diagonal coefficients

    errs_abs = errs_abs.cpu().numpy()
    metrics = {}
    metrics['n_iter'] = n_iter
    metrics['train_time'] = train_time
    metrics['pred_time'] = pred_time
    metrics['loss'] = loss
    metrics['R2'] = R2_list.mean().cpu().numpy()
    metrics['RMSE'] = torch.sqrt(err2.mean()).cpu().numpy()
    metrics['mean_err_abs'], metrics['max_err_abs'] = errs_abs.mean(), errs_abs.max()
    metrics['mean_err_quant05'], metrics['mean_err_quant95'], metrics['mean_err_quant99'] = np.quantile(errs_abs, np.array([0.05, 0.95, 0.99]))
    metrics['mean_sigma'] = std_pred.mean().cpu().numpy()
    metrics['PVA'] = PVA_list.mean().cpu().numpy()
    metrics['alpha_CI'] = alpha_CI.mean().cpu().numpy()  # proportion of points within 2 sigma (should be around 0.95)
    metrics['noise'] = noise_full.cpu().numpy()
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics

##----------------------------------------------------------------------------------------------------------------------
## Training settings
glob_lr_min = 1e-4
glob_lr_max = 1e-1
glob_n_iters = 50000
glob_use_stop = True
glob_loss_thresh = 1e-3 # threshold for loss plateau detection
glob_patience_sched = 1000 # number of iterations without loss decrease before halving the learning rate
glob_patience_crit = glob_patience_sched * 5 # number of iterations without loss decrease before stopping training
##------------------------------------------------------------------------------

def run_models(models_to_run, n_latents, lik_rank, n_ind_points, noise_thresh, X, Y, X_test, Y_test, run_key, results, training_settings=None,
               mean_type=None, kernel_type=None, ker_kwargs=None, stochastic_train_variational_model=False, fix_induc_points_of_var_model=False):
    
    n_points = len(X)
    n_tasks = Y.shape[-1]
    print('Num points: {0}, num tasks: {1}'.format(n_points, n_tasks))
    if n_ind_points is not None:
        train_ind_rat = n_points / n_ind_points
    else:
        train_ind_rat = 1. if fix_induc_points_of_var_model else 1.5

    if training_settings is not None:
        lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched = training_settings
    else:
        lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched = \
        glob_lr_min, glob_lr_max, glob_n_iters, glob_use_stop, glob_loss_thresh, glob_patience_crit, glob_patience_sched
        
    ## Defining models
    kernel_type = gp.kernels.MaternKernel if kernel_type is None else kernel_type
    mean_type = gp.means.ZeroMean if mean_type is None else mean_type
    ker_kwargs = {} if ker_kwargs is None else ker_kwargs
    likelihoods, models, mlls, optimizers, schedulers = {}, {}, {}, {}, {}     

    if 'IGP' in models_to_run:
        models['IGP'] = ExactGPModel(X, Y, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs, batch_lik=False,
                                        lik_mat_rank=lik_rank, noise_thresh=noise_thresh, n_ind_points=n_ind_points)
        likelihoods['IGP'] = models['IGP'].likelihood

    if 'ICM' in models_to_run:
        models['ICM'] = MultitaskGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                        lik_mat_rank=lik_rank, noise_thresh=noise_thresh, n_ind_points=n_ind_points)
        likelihoods['ICM'] = models['ICM'].likelihood

    if 'var' in models_to_run:
        models['var'] = VariationalMultitaskGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                            noise_thresh=noise_thresh, lik_mat_rank=lik_rank,
                            train_ind_ratio=train_ind_rat, seed=0, distrib=gp.variational.CholeskyVariationalDistribution,
                            init_induc_with_qmc=False)
        likelihoods['var'] = models['var'].likelihood
        
    if 'PLMC' in models_to_run:
        models['PLMC'] = ProjectedGPModel(X, Y, n_latents=n_latents, mean_type=mean_type,  kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                        n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=False, diagonal_R=False, scalar_B=False)
        likelihoods['PLMC'] = models['PLMC'].likelihood

    if 'PLMC_fast' in models_to_run:
        models['PLMC_fast'] = ProjectedGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs, 
                                        n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=True, diagonal_R=False, scalar_B=True)
        
        likelihoods['PLMC_fast'] = models['PLMC_fast'].likelihood

    if 'oilmm' in models_to_run:
        models['oilmm'] = ProjectedGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                        n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=True, diagonal_R=True, scalar_B=True, bulk=False)
        likelihoods['oilmm'] = models['oilmm'].likelihood

    ##------------------------------------------------------------------
    ## Configuring optimization
    if gpu:
        X = X.cuda()
        Y = Y.cuda()
        for name in models_to_run:
            models[name] = models[name].cuda()
            likelihoods[name] = likelihoods[name].cuda()

    for name in models_to_run:
        if kernel_type == gp.kernels.SpectralMixtureKernel:
            attribute_string = 'covar_module'
            if name=='ICM':
                attribute_string += '.data_covar_module'
            if n_ind_points is not None and name!='var':
                attribute_string += '.base_kernel'
            attributes = attribute_string.split('.')
            obj = models[name]
            for attr in attributes:
                obj = getattr(obj, attr)
            obj.initialize_from_data_empspect(X, Y.mT)  # Spectral Mixture Kernel has to be carefully initialized
        models[name].train()
        likelihoods[name].train()
        mlls[name] = models[name].default_mll()
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=lr_max)
        # schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizers[name], lambda_lr)
        schedulers[name] = CorrectReduceLROnPlateau(optimizers[name], factor=0.5, patience=patience_sched,
                              threshold=loss_thresh, threshold_mode='rel', mode='min', min_lr=lr_min)

    ##------------------------------------------------------------------
    ## Training models

    times, last_losses = {}, {}
    effective_n_iters = {model_name : n_iters for model_name in models_to_run}
    # wandb.init(project="realdata_experiment", config={"learning_rate": 0.01, "epochs": n_iters})
    for name in models_to_run:
        print(' \n Training {0} model ... \n'.format(name))
        start = time.time()
        best_loss = 1e9
        last_lr = 1.
        no_improve_count = 0
        if name == "var" and stochastic_train_variational_model:
            train_dataset = TensorDataset(X, Y.mT)
            train_loader = DataLoader(train_dataset, batch_size=n//10, shuffle=False)
        for i in range(n_iters):
            if name == "var" and stochastic_train_variational_model:
                new_loss = 0.
                for X_batch, Y_batch in train_loader:
                    Y_batch = Y_batch.mT
                    optimizers[name].zero_grad()
                    with gp.settings.cholesky_max_tries(8):
                        output_train = models[name](X_batch)
                        loss = -mlls[name](output_train, Y_batch)
                        loss.backward()
                        optimizers[name].step()
                    new_loss += loss.item()
                new_loss /= train_loader.batch_size
            else:
                optimizers[name].zero_grad()
                with gp.settings.cholesky_max_tries(8):
                    output_train = models[name](X)
                    loss = -mlls[name](output_train, Y)
                    loss.backward()
                    optimizers[name].step()
                    new_loss = loss.item()

            if print_loss and i%freq_print==0:
                print(new_loss)
            schedulers[name].step(new_loss)
            current_lr = optimizers[name].param_groups[0]['lr']
            if current_lr < last_lr:
                print('LR changed to {0}'.format(current_lr))
                last_lr = current_lr
                no_improve_count = 0

            loss_better_than_best = (new_loss >= 0. and new_loss < best_loss * (1 - loss_thresh)) \
                                    or (new_loss < 0. and new_loss < best_loss * (1 + loss_thresh))
            loss_almost_as_good_as_best = (new_loss >= 0. and new_loss < best_loss * (1 + loss_thresh)) \
                                    or (new_loss < 0. and new_loss < best_loss * (1 - loss_thresh))

            if loss_better_than_best :
                best_loss = new_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            # wandb.log({"loss": new_loss, "best_loss": best_loss, "lr": current_lr, "iteration":i})
            last_losses[name] = new_loss
            if use_stop and (no_improve_count > patience_crit) and loss_almost_as_good_as_best:
                effective_n_iters[name] = i
                break

        times[name] = time.time() - start

    ##------------------------------------------------------------------
    ## Making predictions

    for name in models_to_run:
        models[name].eval()
        likelihoods[name].eval()
        if gpu:
            models[name] = models[name].cpu()
            likelihoods[name] = likelihoods[name].cpu()

    for name in models_to_run:
    # all these algebra options have been tested to have little impact on results
        with torch.no_grad(),\
            gp.settings.eval_cg_tolerance(1e-2),\
            gp.settings.cholesky_max_tries(8):

            print(' \n Making predictions for {0} model...'.format(name))
            start = time.time()
            if hasattr(models[name], 'full_likelihood'):  # we have to compute the full likelihood of projected models
                full_likelihood = models[name].full_likelihood()
            else:
                full_likelihood = likelihoods[name]

            observed_pred = full_likelihood(models[name](X_test))
            pred_y = observed_pred.mean
            var_pred = observed_pred.variance
            std_pred = var_pred.sqrt().squeeze()
            pred_time = time.time() - start

            if hasattr(full_likelihood, 'task_noise_covar'):
                Sigma_guess = full_likelihood.task_noise_covar
            else:
                Sigma_guess = torch.diag_embed(full_likelihood.task_noises)
            ##------------------------------------------------------------------
            ## Computing, displaying and storing performance metrics
            metrics = compute_metrics(y_test=Y_test, y_pred=pred_y, std_pred=std_pred, loss=last_losses[name], Sigma_guess=Sigma_guess,
                                    n_iter=effective_n_iters[name], train_time=times[name], pred_time=pred_time, print_metrics=print_metrics)
            metrics.update(v)
            metrics['model'] = name
            metrics['eps'] = models[name].current_eps if hasattr(models[name], 'current_eps') else 0.
            results[name + run_key] = metrics
    return results, models

##----------------------------------------------
def detrend_data(x, y, degree=1):
    coef = np.polyfit(x, y, degree)
    return y - np.polyval(coef, x)

## Tidal height experiment
if experiment=='tidal':
    import os
    from scipy.interpolate import interp1d
    from datetime import datetime
    torch.set_default_dtype(torch.float32)

    ## Data preprocessing
    root = '_experiments/bramblemet/'
    degree = 2 # degree of the polynomial detrending
    ndiv = 4 # subsampling factor
    start_date = '2020-06-01'
    end_date = '2020-06-16'
    dico = {}
    stations = ['bramblemet', 'cambermet', 'chimet', 'sotonmet']
    for station in stations:
        df = pd.read_csv(os.path.join(root,'{0}.csv.gz'.format(station)), compression='gzip', low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
        df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
        df['time_num'] = df['Date'].apply(lambda x: x.timestamp())
        values = df['DEPTH'].values
        if 'time_num' not in dico:  # create a reference time vector with values between 0 and 1
            ref_time = df['time_num'].values
            ref_time_norm = ref_time / ref_time.max()
            ref_time_norm = ref_time_norm - ref_time_norm[0]
            dico['time_num'] = ref_time_norm
            dico['Date'] = df['Date'].values
        else:
            f = interp1d(df['time_num'].values, values) # align all time series on the same reference time vector
            values = f(ref_time)
        dico[station] = detrend_data(ref_time_norm, values, degree=degree)
    df = pd.DataFrame(dico).set_index('Date').astype(np.float32)
    
    ## Data formatting for model use
    df = df.iloc[::ndiv] # subsampling the time series by a factor ndiv
    X, Y = df['time_num'].values[:,None], df.drop('time_num', axis=1).values
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    num_days = (end_date - start_date).days
    test_indices = np.arange(len(df)//num_days) # test set is one day in the middle of the time series

    Y = Y[:, :1]

    X, X_test = np.delete(X, test_indices, axis=0), X[test_indices]
    Y, Y_test = np.delete(Y, test_indices, axis=0), Y[test_indices]
    Mean, Std = Y.mean(axis=0), Y.std(axis=0)
    Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
    n_points, n_tasks = Y.shape
    X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)

    ## Model parameters
    kernel_type = gp.kernels.SpectralMixtureKernel
    v = {
        'q': 2, 
        'lik_rank': 0,
        'n_mix':2,
        'void' : [0.]}
    v_vals = {
        'q' : range(1, n_tasks+1), 
        'lik_rank' : [0, n_tasks],
        'n_mix': range(2,10),
        'void' : [0.]}
    v_test_0 = 'n_mix'
    v_test_1 = 'void'
    noise_thresh = 1e-3
    n_ind_points = None

    ## Specific training settings
    spec_lr_min = 1e-4
    spec_lr_max = 1e-2
    spec_n_iters = 50000
    spec_use_stop = True
    spec_loss_thresh = 1e-3
    spec_patience_sched = 1000
    spec_patience_crit = spec_patience_sched * 5
    training_settings = (spec_lr_min, spec_lr_max, spec_n_iters, spec_use_stop, spec_loss_thresh, spec_patience_crit, spec_patience_sched)

    appendix += 'div{0}_{1}days'.format(ndiv, num_days)
    if n_ind_points is not None:
        appendix += '_{0}ind'.format(n_ind_points)
    appendix += 'test' # to further customize experiment name
    path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test_0 + '_'+ v_test_1 + '_' + v_test_2 + '.csv'
    print(path + '\n')
    results = {}
    min_err = 1.
    
    for i_v, vval in enumerate(v_vals[v_test_0]):
        for i_v1, vval1 in enumerate(v_vals[v_test_1]):
            for i_v2, vval2 in enumerate(v_vals[v_test_2]):
                v[v_test_0] = vval
                v[v_test_1] = vval1
                v[v_test_2] = vval2
                q, lik_rank = v['q'], v['lik_rank']
                run_key = v_test_0 + '_' + v_test_1 + '_' + v_test_2 + '_{0}_{1}_{2}'.format(i_v, i_v1, i_v2)
                results, models = run_models(models_to_run=models_to_run,
                                                n_latents=q,
                                                lik_rank=lik_rank,
                                                n_ind_points=n_ind_points,
                                                noise_thresh=noise_thresh,
                                                X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                                                run_key=run_key, results=results,
                                                kernel_type=kernel_type,
                                                ker_kwargs={'num_mixtures':v['n_mix']},
                                                fix_induc_points_of_var_model=True,
                                                training_settings=training_settings,
                                                )
                for model in models_to_run: # we keep the best model for illustrating predictions (Figure7 in the paper) 
                    if results[model + run_key]['RMSE']< min_err:
                        best_model = models[model]
                        min_err = results[model + run_key]['RMSE']

    res_df = pd.DataFrame.from_dict(results, orient='index')
    if export_results:
        res_df.to_csv(path)

    ## Illustrating predictions
    df[['pred'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
    df[['lower'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
    df[['upper'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
    best_model.eval()
    best_model.likelihood.eval()
    full_likelihood = best_model.full_likelihood() if hasattr(best_model, 'full_likelihood') else best_model.likelihood
    observed_pred = full_likelihood(best_model(X_test))
    pred_y = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
    test_indices = df.index[test_indices]
    for i in range(n_tasks):
        df.loc[test_indices, 'pred'+str(i)] = pred_y[:,i].detach().cpu().numpy()
        df.loc[test_indices, 'lower'+str(i)] = lower[:,i].detach().cpu().numpy()
        df.loc[test_indices, 'upper'+str(i)] = upper[:,i].detach().cpu().numpy()
        df.to_csv(path.replace('realdata_study', 'preds'))

    df = df.reset_index()
    df = df[(df['Date'] <= '2020-06-07')]
    df['Date'] = pd.to_datetime(df['Date'])
    test_indices = np.where(df['pred0']!=0.)[0]
    sub_indices = np.arange(0, test_indices[0])
    sup_indices = np.arange(test_indices[-1]+1, len(df))
    df_sub, df_sup = df.iloc[sub_indices, :], df.iloc[sup_indices, :]
    test_indices = df.index.values[test_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df_sub['Date'], df_sub['bramblemet'], color='blue')
    ax.scatter(df_sub['Date'], df_sub['bramblemet'], label='training data', color='blue', marker='.')
    ax.plot(df_sup['Date'], df_sup['bramblemet'], color='blue')
    ax.scatter(df_sup['Date'], df_sup['bramblemet'], color='blue', marker='.')
    ax.scatter(df.loc[test_indices, 'Date'], df.loc[test_indices, 'bramblemet'], label='test data', color='k', marker='x')
    ax.plot(df.loc[test_indices, 'Date'], df.loc[test_indices, 'pred0'], color='red', label='prediction')
    ax.fill_between(df.loc[test_indices, 'Date'], df.loc[test_indices, 'lower0'], df.loc[test_indices, 'upper0'], color='red', alpha=0.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Tide height (m)')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend()
    plt.savefig('bramblemet_test')
    plt.close()


##----------------------------------------------
## Ship
if experiment=='ship':
    torch.set_default_dtype(torch.float32)

    ## Data preprocessing
    root = '_experiments/ship/'
    ndiv = 5 # subsampling factor
    data = pd.read_csv(root + "data.txt", sep=r"\s+", engine="python", dtype=str, header=None).astype(np.float32)
    data = data.iloc[::ndiv]
    X = data.iloc[:, [0, 16, 17]].values
    Y = data.drop([0, 1, 8, 11, 16, 17], axis=1).values

    ## Data formatting for model use
    X, X_test = X[:-100], X[-100:]
    Y, Y_test = Y[:-100], Y[-100:]
    Mean, Std = Y.mean(axis=0), Y.std(axis=0)
    Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
    n_points, n_tasks = Y.shape
    X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)
    Y = Y.mT

    ## Model parameters
    v = {
        'q': 3, 
        'lik_rank': 0,
        'void' : [0.]}
    v_vals = {
        'q' : range(1,11), 
        'lik_rank' : [0,n_tasks],
        'void' : [0.]}
    v_test = 'void'
    n_ind_points = 100
    noise_thresh = 1e-3

    ## Specific training settings
    spec_lr_min = 1e-4
    spec_lr_max = 1e-1
    spec_n_iters = 100000
    spec_use_stop = True
    spec_loss_thresh = 1e-4
    spec_patience_sched = 1000
    spec_patience_crit = spec_patience_sched * 5
    training_settings = (spec_lr_min, spec_lr_max, spec_n_iters, spec_use_stop, spec_loss_thresh, spec_patience_crit, spec_patience_sched)

    appendix += 'div{0}'.format(ndiv)
    if n_ind_points is not None:
        appendix += '_{0}ind'.format(n_ind_points)
    appendix += 'v2'
    path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test + '_'+ v_test_2 + '.csv'
    print(path + '\n')

    results = {}
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v[v_test] = vval
            v[v_test_2] = vval2
            q, lik_rank = v['q'], v['lik_rank']
            run_key = v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
            results, models = run_models(models_to_run=models_to_run,
                                                n_latents=q,
                                                lik_rank=lik_rank,
                                                n_ind_points=n_ind_points,
                                                noise_thresh=noise_thresh,
                                                X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                                                run_key=run_key, results=results,
                                                )
    df = pd.DataFrame.from_dict(results, orient='index')
    if export_results:
        df.to_csv(path)

##----------------------------------------------
## Neutronics
if experiment=='neutro':
    torch.set_default_dtype(torch.float32)

    ## Data formatting for model use
    root = '../data/'
    X = torch.load(root + 'train_x_sobol256.pt')
    X_test = torch.load(root + 'test_x_lhs512.pt')
    Y = torch.load(root + 'train_y_sobol256.pt')
    Y_test = torch.load(root + 'test_y_lhs512.pt')
    n_points, n_tasks = Y.shape
    Y = Y.mT

    ## Model parameters
    v = {
        'n_lat': 16, 
        'lik_rank': 0,
        'void' : [0.]}
    v_vals = {
        'n_lat' : range(1,n_tasks+1), 
        'lik_rank' : [0, n_tasks],
        'void' : [0.]}
    v_test = 'void'
    n_ind_points = None
    noise_thresh = 1e-4

    if n_ind_points is not None:
        appendix += '{0}ind'.format(n_ind_points)
    appendix += ''
    path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test + '_'+ v_test_2 + '.csv'
    print(path + '\n')

    results = {}
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v[v_test] = vval
            v[v_test_2] = vval2
            n_lat, lik_rank = v['n_lat'], v['lik_rank']
            run_key = v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
            results, models = run_models(models_to_run=models_to_run,
                                                n_latents=q,
                                                lik_rank=lik_rank,
                                                n_ind_points=n_ind_points,
                                                noise_thresh=noise_thresh,
                                                X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                                                run_key=run_key, results=results,
                                                )
    df = pd.DataFrame.from_dict(results, orient='index')
    if export_results:
        df.to_csv(path)

##----------------------------------------------
## SARCOS
if experiment=='sarcos':
    from scipy.io import loadmat
    torch.set_default_dtype(torch.float32)

    ## Data preprocessing
    ndiv = 10 # subsampling factor
    root = '_experiments/SARCOS/'
    train_data = loadmat(root + 'sarcos_inv.mat')['sarcos_inv'].astype(np.float32)[::ndiv,:]
    test_data = loadmat(root + 'sarcos_inv_test.mat')['sarcos_inv_test'].astype(np.float32)

    ## Data formatting for model training
    X, Y = train_data[:, :21], train_data[:, 21:]
    X_test, Y_test = test_data[:, :21], test_data[:, 21:]
    Mean, Std = Y.mean(axis=0), Y.std(axis=0)
    Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
    n_points, n_tasks = Y.shape
    X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)
    Y = Y.mT

    ## Model parameters
    v = {
        'q': n_tasks, 
        'lik_rank': 0,
        'void' : [0.]}
    v_vals = {
        'q' : range(1, n_tasks+1), 
        'lik_rank' : [0, n_tasks],
        'void' : [0.]}
    v_test = 'void'
    n_ind_points = 500
    noise_thresh = 1e-3

    appendix += 'div{0}'.format(ndiv)
    if n_ind_points is not None:
        appendix += '_{0}ind'.format(n_ind_points)
    path = 'results/realdata_study_'+ experiment + '_' + appendix + '_' + v_test + '_'+ v_test_2 + '.csv'
    print(path + '\n')

    results = {}
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v[v_test] = vval
            v[v_test_2] = vval2
            q, lik_rank = v['q'], v['lik_rank']
            run_key = v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
            results, models = run_models(models_to_run=models_to_run,
                                                n_latents=q,
                                                lik_rank=lik_rank,
                                                n_ind_points=n_ind_points,
                                                noise_thresh=noise_thresh,
                                                X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                                                run_key=run_key, results=results,
                                                )
    df = pd.DataFrame.from_dict(results, orient='index')
    if export_results:
        df.to_csv(path)
