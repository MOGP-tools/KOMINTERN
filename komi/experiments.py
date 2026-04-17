
import os
import time

import numpy as np
import torch
gpu = True
if gpu:
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float32)
from torch.utils.data import TensorDataset, DataLoader
import gpytorch as gp
import pandas as pd
# import wandb
# import matplotlib.pyplot as plt

from mogp_icm import MultitaskGPModel
from mogp_var import VariationalMultitaskGPModel
from mogp_plmc import ProjectedGPModel
from base_gp import ExactGPModel

from utilities import CorrectReduceLROnPlateau

##----------------------------------------------------------------------------------------------------------------------
## Setting default parameters

v = {  # default parameter values
'n' : 500, #500,
'p' : 100, #100,
'q' : 10, #10,
'q_guess' : 10,  # q_guess is here to investigate model misspecification (q is the number of latent processes of the data)
'q_noise' : 10, #10,
'q_noise_guess' : 10,  # q_noise_guess is here to investigate model misspecification (q_noise is the number of latent processes of the data)
'mu_noise' : 1e-1, #1e-1,
'mu_str' : 0.5, #0.5,
'min_scale' : 0.01, #0.1,
'void' : 0.
}

v_vals = {  # values to be tested
'n' : range(100, 501, 50),
'p' : range(25, 201, 25),
'q' : range(10, 91, 10),
'q_guess' : range(1, 32, 10),
'q_noise' : range(10, 91, 10),
'q_noise_guess' : range(0, 91, 10),
'mu_noise' : np.logspace(-3, np.log10(0.5), 10),
'mu_str' : np.linspace(0., 1., 10),
'min_scale' : np.linspace(0.1, 1., 10),
'void' : [0.]
}
max_scale = 1.,
n_test = 2500 # number of test points

models_to_run = ['ICM','PLMC','oilmm','var','PLMC_fast']
models_to_run = ['PLMC_fast']
v_test = 'void' # replace by the parameter to be tested (or perform a one-point experiment)
v_test_2 = 'void' # if not 'void', interaction between two parameters is tested
n_random_runs = 1 # number of random repetitions of the experiment (each one with different data)
##----------------------------------------------------------------------------------------------------------------------

## >>> Reproducing the experiments from the paper : just uncomment the desired line and run the script <<<

# v_test, n_random_runs = 'mu_noise', 50
# v_test, n_random_runs = 'mu_str', 50
# v_test, n_random_runs = 'p', 50
# v_test, n_random_runs = 'q', 50
# v_test, n_random_runs = 'q_noise', 50
# v_test, n_random_runs = 'min_scale', 50
# v_test, n_random_runs = 'q_noise_guess', 50
# v_test, n_random_runs = 'n', 50
##----------------------------------------------------------------------------------------------------------------------

## Results formatting
reprise_csv = False
print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000
appendix = '' # to further customize experiment name
path = 'results/parameter_study_' + v_test + '_' + v_test_2 + appendix + '.csv'
export_results = True

##----------------------------------------------------------------------------------------------------------------------
## Training settings
lr_min = 1e-4
lr_max = 1e-1
n_iters = 100000
use_stop = True
loss_thresh = 1e-3 # threshold for loss plateau detection
patience_sched = 1000 # number of iterations without loss improvement before halving the learning rate
patience_crit = patience_sched * 5 # number of iterations without loss improvement before stopping training
stochastic_train_variational_model = False
##------------------------------------------------------------------------------
def compute_metrics(y_test, y_pred, std_pred, loss, Sigma_true, Sigma_guess, n_iter, train_time, pred_time, print_metrics=True):
    delta = y_test - y_pred
    errs_abs = torch.abs(delta).squeeze()
    alpha_CI = torch.mean((errs_abs < 2 * std_pred).float())
    err2 = errs_abs ** 2
    R2_list = 1 - torch.mean(err2, dim=0) / torch.var(y_test, dim=0)
    PVA_list = torch.log(torch.mean(err2 / std_pred ** 2, dim=0))
    noise_full = Sigma_guess.diag().mean() # mean of the diagonal coefficients
    Sigma_diff = torch.abs(Sigma_guess - Sigma_true)

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
    metrics['noise_true'] = Sigma_true.diag().mean().cpu().numpy()
    metrics['noise_mat_err'] = (torch.linalg.norm(Sigma_diff) / torch.linalg.norm(Sigma_true)).cpu().numpy()
    metrics['noise_diag_err'] = (torch.linalg.norm(Sigma_diff.diag() / Sigma_true.diag())).cpu().numpy()
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics

##------------------------------------------------------------------------------
## Running experiments

if reprise_csv and os.path.isfile(path):
    stored_df = pd.read_csv(path, index_col=0)
else:
    stored_df = None

# wandb.init(project="my-torch-project", config={"learning_rate": 0.01, "epochs": n_iters})

for i_run in range(n_random_runs):
    print('\n Random run number {} : \n'.format(i_run))
    np.random.seed(i_run)
    torch.manual_seed(i_run)
    results = {}
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v[v_test] = vval
            v[v_test_2] = vval2
            p, q, q_noise, n, mu_noise, mu_str, min_scale, q_noise_guess = \
                v['p'], v['q'], v['q_noise'], v['n'], v['mu_noise'], v['mu_str'], v['min_scale'], v['q_noise_guess']
            run_key = v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
            ##------------------------------------------------------------------

            ## Generating artificial data
            lscales = torch.as_tensor(np.linspace(min_scale, max_scale, q))
            ker_list = [gp.kernels.MaternKernel() for i in range(q)]
            for i in range(q):
                ker_list[i].lengthscale = lscales[i]

            X_train = torch.linspace(-1, 1, n)
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            X_test = 2*torch.rand(n_test) - 1
            X = torch.cat([X_train, X_test], dim=0)
            H_true = torch.randn(size=(q, p))
            lat_gp_dist = [gp.distributions.MultivariateNormal(torch.zeros_like(X), kernel(X)) for kernel in ker_list]
            gp_vals = torch.stack([dist.sample() for dist in lat_gp_dist])
            Y_sig = gp_vals.T @ H_true * (1 - mu_noise)

            ## structured noise
            H_noise_true = torch.randn(size=(q_noise, p))
            gp_vals_hid_com = torch.randn((n + n_test, q_noise))
            Y_noise_com = gp_vals_hid_com @ H_noise_true * mu_str

            ## unstructured noise
            noise_levels = torch.rand(p) + 0.1
            gp_vals_hid_spec = noise_levels[None, :] * torch.randn((n + n_test, p)) # homosk noise
            Y_noise_spec = gp_vals_hid_spec * (1 - mu_str)

            Y_noise = (Y_noise_com + Y_noise_spec) * mu_noise
            Sigma_true = (H_noise_true.T @ H_noise_true * mu_str**2 + torch.diag_embed(noise_levels**2) * (1 - mu_str)**2) * mu_noise**2
            Y = Y_sig + Y_noise
            X = X[:,None]
            X_test, Y_test = X[n:], Y[n:]
            X, Y = X[:n], Y[:n]
            ##------------------------------------------------------------------

            ## Defining models
            kernel_type = gp.kernels.MaternKernel
            mean_type = gp.means.ZeroMean
            noise_thresh = 1e-3
            train_ind_rat = 1. # ratio between number of training points and number of inducing points for the variational model
            n_ind_points = None  # if not None, all models will use an inducing point approximation with ths number of inducing points

            if v_test!= 'q_noise_guess':  # if q_noise_guess is not the parameter to be tested, we use a full rank noise (general Sigma matrix)
                q_noise_guess, v['q_noise_guess'] = p, p
            if v_test!= 'q_guess':  # if q_guess is not the parameter to be tested, we use a full rank noise (general Sigma matrix)
                q_guess, v['q_guess'] = q, q
            # q_noise_guess = 0

            likelihoods, models, mlls, optimizers, schedulers = {}, {}, {}, {}, {}
            
            if 'ICM' in models_to_run:
                models['ICM'] = MultitaskGPModel(X, Y, n_latents=q_guess, mean_type=mean_type, kernel_type=kernel_type,
                                                lik_mat_rank=q_noise_guess, noise_thresh=noise_thresh, n_ind_points=n_ind_points)
                likelihoods['ICM'] = models['ICM'].likelihood

            if 'var' in models_to_run:
                models['var'] = VariationalMultitaskGPModel(X, Y, n_latents=q_guess, mean_type=mean_type, kernel_type=kernel_type,
                                    noise_thresh=noise_thresh, lik_mat_rank=q_noise_guess,
                                    train_ind_ratio=train_ind_rat, seed=0, distrib=gp.variational.CholeskyVariationalDistribution,
                                    init_induc_with_qmc=False)
                likelihoods['var'] = models['var'].likelihood
                
            if 'PLMC' in models_to_run:
                models['PLMC'] = ProjectedGPModel(X, Y, n_latents=q_guess, mean_type=mean_type,  kernel_type=kernel_type, diagonal_B=False,
                                                n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=False, diagonal_R=False, scalar_B=False)
                likelihoods['PLMC'] = models['PLMC'].likelihood

            if 'PLMC_orthpar' in models_to_run:
                models['PLMC_orthpar'] = ProjectedGPModel(X, Y, n_latents=q_guess, mean_type=mean_type,  kernel_type=kernel_type, bulk=False, 
                                                n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=False, diagonal_R=False, scalar_B=False)
                likelihoods['PLMC_orthpar'] = models['PLMC_orthpar'].likelihood

            if 'PLMC_fast' in models_to_run:
                models['PLMC_fast'] = ProjectedGPModel(X, Y, n_latents=q_guess, mean_type=mean_type, kernel_type=kernel_type, 
                                                n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=True, diagonal_R=False, scalar_B=True)
                likelihoods['PLMC_fast'] = models['PLMC_fast'].likelihood

            if 'oilmm' in models_to_run:
                models['oilmm'] = ProjectedGPModel(X, Y, n_latents=q_guess, mean_type=mean_type, kernel_type=kernel_type, 
                                                n_ind_points=n_ind_points, noise_thresh=noise_thresh, BDN=True, diagonal_R=True, scalar_B=True, bulk=False)
                likelihoods['oilmm'] = models['oilmm'].likelihood

            ##------------------------------------------------------------------
                
            ## Configuring optimization
            if gpu:
                X = X.cuda()
                Y = Y.cuda()
                Sigma_true = Sigma_true.cuda()
                for name in models_to_run:
                    models[name] = models[name].cuda()
                    likelihoods[name] = likelihoods[name].cuda()

            for name in models_to_run:
                models[name].train()
                likelihoods[name].train()
                mlls[name] = models[name].default_mll()
                optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=lr_max)
                schedulers[name] = CorrectReduceLROnPlateau(optimizers[name], factor=0.5, patience=patience_sched, mode='min',
                                                                                threshold=loss_thresh, threshold_mode='rel', min_lr=lr_min)
            ##------------------------------------------------------------------
            
            ## Training models                 
            times, last_losses = {}, {}
            effective_n_iters = {model_name : n_iters for model_name in models_to_run}
            for name in models_to_run:
                print(' \n Training {0} model ... \n'.format(name))
                start = time.time()
                best_loss = 1e9
                last_lr = 1.
                if name == "var" and stochastic_train_variational_model:
                    train_dataset = TensorDataset(X, Y.mT)
                    train_loader = DataLoader(train_dataset, batch_size=n//10, shuffle=False)
                for i in range(n_iters):
                    if name == "var" and stochastic_train_variational_model:
                        new_loss = 0.
                        for X_batch, Y_batch in train_loader:
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
                    X_test, Y_test = X_test.cuda(), Y_test.cuda()

            for name in models_to_run:
            # The skip_posterior_variances option is here to be able to compute posterior mean for model ICM, even when
            # covariance computation would saturate memory. It should be deactivated when possible.
                skip_var = (name=='ICM')
                with torch.no_grad(),\
                    gp.settings.skip_posterior_variances(state=skip_var), \
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
                    if skip_var:
                        var_pred = torch.ones_like(pred_y)
                    else:
                        var_pred = observed_pred.variance
                    std_pred = var_pred.sqrt().squeeze()
                    pred_time = time.time() - start

                    if hasattr(full_likelihood, 'task_noise_covar'):
                        Sigma_guess = full_likelihood.task_noise_covar
                    else:
                        Sigma_guess = torch.diag_embed(full_likelihood.task_noises)

                    ##------------------------------------------------------------------
                    ## Computing, displaying and storing performance metrics
                    metrics = compute_metrics(y_test=Y_test, y_pred=pred_y, std_pred=std_pred, loss=last_losses[name],
                                                n_iter=effective_n_iters[name], train_time=times[name], pred_time=pred_time,
                                                Sigma_guess=Sigma_guess, Sigma_true=Sigma_true, print_metrics=print_metrics)
                    metrics.update(v)
                    metrics['model'] = name
                    metrics['run_key'] = run_key
                    full_run_key = name + '_' + run_key
                    metrics['full_run_key'] = full_run_key
                    results[str(i_run) + '_' + full_run_key] = metrics
    ##------------------------------------------------------------------
    ## Exporting results
    print(path + '\n')
    new_df = pd.DataFrame.from_dict(results, orient='index')
    if stored_df is None:
        stored_df = new_df
    else:
        stored_df = pd.concat([stored_df, new_df])
    stored_df.to_csv(path)
