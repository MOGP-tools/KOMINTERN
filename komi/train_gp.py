import time
import pickle
import json
import zipfile

import numpy as np
np.random.seed(seed=12)
import pandas as pd

import torch
torch.manual_seed(12)
torch.backends.cuda.matmul.allow_tf32 = False
import gpytorch as gp

from custom_profiler import profiler, magic_profiler # info : https://github.com/KarGeekrie/customProfiler

def testHelloWorld():
    print("Hello")

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


def compute_metrics(y_test, y_pred, sigma_pred, loss, H_guess_hid, n_iter, train_time, pred_time, print_metrics=True, test_mask=None):
    if test_mask is not None:  # this can be used to compute metrics on a subset of outputs
        y_test = y_test[test_mask]
        y_pred = y_pred[test_mask]
        sigma_pred = sigma_pred[test_mask]
    delta = y_test - y_pred
    errs_abs = torch.abs(delta).squeeze()
    alpha_CI = torch.mean((errs_abs < 2 * sigma_pred).float())
    err2 = errs_abs ** 2
    R2_list = 1 - torch.mean(err2, dim=0) / torch.var(y_test, dim=0)
    PVA_list = torch.log(torch.mean(err2 / sigma_pred ** 2, dim=0))
    noise_full = (H_guess_hid**2).sum() / y_test.shape[1] # mean of the diagonal coefficients

    errs_abs = errs_abs.cpu().numpy()
    metrics = {}
    metrics['n_iter'] = n_iter
    metrics['train_time'] = train_time
    metrics['pred_time'] = pred_time
    metrics['loss'] = loss
    metrics['noise'] = noise_full.cpu().numpy()
    metrics['R2'] = R2_list.mean().cpu().numpy()
    metrics['RMSE'] = torch.sqrt(err2.mean()).cpu().numpy()
    metrics['mean_err_abs'], metrics['max_err_abs'] = errs_abs.mean(), errs_abs.max()
    metrics['mean_err_quant05'], metrics['mean_err_quant95'], metrics['mean_err_quant99'] = np.quantile(errs_abs, np.array([0.05, 0.95, 0.99]))
    metrics['mean_sigma'] = sigma_pred.mean().cpu().numpy()
    metrics['PVA'] = PVA_list.mean().cpu().numpy()
    metrics['alpha_CI'] = alpha_CI.mean().cpu().numpy()
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics

def run_models(models_to_run, models_with_sched, q, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                n_iters, lr_min, loss_thresh, patience, print_metrics, print_loss, freq_print, gpu, 
                train_ind_rat, n_ind_points, run_key, results, test_mask=None, renorm_func=None, mean_type=None,
                kernel_type=None, decomp=None, lambda_lr=None, stopping_crit='mean', ker_kwargs={}):
    
    ## Defining models
    kernel = gp.kernels.MaternKernel if kernel_type is None else kernel_type
    Mean = gp.means.ZeroMean if mean_type is None else mean_type
    likelihoods, models, mlls, optimizers, schedulers = {}, {}, {}, {}, {}
      
    if 'ICM' in models_to_run:
        likelihoods['ICM'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank, has_global_noise=True)
        models['ICM'] = MultitaskGPModel(X, Y, likelihoods['ICM'], n_tasks=n_tasks, init_lmc_coeffs=True,
                            n_latents=q, mean_type=Mean, kernel_type=kernel, decomp=decomp, n_inducing_points=n_ind_points, 
                            fix_diagonal=False, model_type='ICM', ker_kwargs=ker_kwargs)

    if 'var' in models_to_run:
        likelihoods['var'] = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=lik_rank)
        TI_rat = train_ind_rat if n_ind_points is None else n_points / n_ind_points
        models['var'] = VariationalMultitaskGPModel(X, train_y=Y, n_tasks=n_tasks, init_lmc_coeffs=True,
                            mean_type=Mean, kernel_type=kernel, n_latents=q, decomp=decomp,
                            train_ind_ratio=TI_rat, seed=0, distrib=gp.variational.CholeskyVariationalDistribution,
                            var_strat=gp.variational.UnwhitenedVariationalStrategy,
                            ker_kwargs=ker_kwargs)

    if 'proj' in models_to_run:
        models['proj'] = ProjectedGPModel(X, Y, n_tasks, q, proj_likelihood=None, noise_thresh=-9,
                                        mean_type=Mean, kernel_type=kernel, decomp=decomp, bulk=True,
                                        init_lmc_coeffs=True, BDN=True, diagonal_B=True, diagonal_R=False,
                                        scalar_B=True, ker_kwargs=ker_kwargs, n_inducing_points=n_ind_points)
        likelihoods['proj'] = models['proj'].likelihood

    ##------------------------------------------------------------------
    ## Configuring optimization
    if gpu:
        X = X.cuda()
        Y = Y.cuda()
        for name in models_to_run:
            models[name] = models[name].cuda()
            likelihoods[name] = likelihoods[name].cuda()

    for name in models_to_run:
        models[name].train()
        likelihoods[name].train()

    if 'ICM' in models_to_run:
        mlls['ICM'] = gp.mlls.ExactMarginalLogLikelihood(likelihoods['ICM'], models['ICM'])
        optimizers['ICM'] = torch.optim.AdamW(models['ICM'].parameters(), lr=lrs['ICM'])  # Includes GaussianLikelihood parameters
    if 'var' in models_to_run:
        mlls['var'] = gp.mlls.VariationalELBO(likelihoods['var'], models['var'], num_data=n_points)
        optimizers['var'] = torch.optim.AdamW([{'params': models['var'].parameters()}, {'params': likelihoods['var'].parameters()}], lr=lrs['var'])
    if 'proj' in models_to_run:
        mlls['proj'] = ProjectedLMCmll(likelihoods['proj'], models['proj'])
        optimizers['proj'] = torch.optim.AdamW(models['proj'].parameters(), lr=lrs['proj'])

    for name in models_to_run:
        if name in models_with_sched:
            if lambda_lr is None:
                schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(optimizers[name], gamma=np.exp(np.log(lr_min / lrs[name]) / n_iters[name]))
            else:
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizers[name], lambda_lr)

    ##------------------------------------------------------------------
    ## Training models

    times, last_losses = {}, {}
    effective_n_iters = n_iters.copy()
    for name in models_to_run:
        if name not in mods_to_train:
            continue
        print(' \n Training {0} model ... \n'.format(name))
        start = time.time()
        plateau_id = 0
        last_loss = 1e-9
        deltas = np.zeros(patience)
        for i in range(n_iters[name]):
            optimizers[name].zero_grad()
            with gp.settings.cholesky_jitter(1e-5):
                output_train = models[name](X)
                loss = -mlls[name](output_train, Y)
                if print_loss and i%freq_print==0:
                    print(loss.item())
                loss.backward()
                optimizers[name].step()
            if name in schedulers:
                schedulers[name].step()

            new_loss = loss.item()
            if stopping_crit == 'max':
                if i>0 and np.abs( 1 - new_loss / last_losses[name]) < loss_thresh:
                    plateau_id += 1
                    if plateau_id > patience :
                        effective_n_iters[name] = i
                        break
                else:
                    plateau_id = 0
            elif stopping_crit == 'mean':
                for j in range(patience - 1):
                    deltas[j+1] = deltas[j]
                deltas[0] = np.abs( 1 - new_loss / last_loss)
                if i >= patience and deltas.mean() < loss_thresh:
                    effective_n_iters[name] = i
                    break

            last_losses[name] = new_loss

        times[name] = time.time() - start

    ##------------------------------------------------------------------
    ## Making predictions

    for name in models_to_run:
        models[name].eval()
        likelihoods[name].eval()
        if gpu:
            models[name] = models[name].cpu()
            likelihoods[name] = likelihoods[name].cpu()

    sigma, preds_y = {}, {}
    for name in models_to_run:
    # all these algebra options have been tested to have little impact on results. Warning on the skip_posterior_variances option !!
        skip_var = (name=='ICM')
        with torch.no_grad(),\
                    gp.settings.skip_posterior_variances(state=skip_var), \
                    gp.settings.skip_logdet_forward(state=False), \
                    gp.settings.cg_tolerance(1e-0),\
                    gp.settings.eval_cg_tolerance(1e-2),\
                    gp.settings.max_cholesky_size(128), \
                    gp.settings.max_lanczos_quadrature_iterations(20), \
                    gp.settings.max_preconditioner_size(15), \
                    gp.settings.max_root_decomposition_size(100), \
                    gp.settings.min_preconditioning_size(2000), \
                    gp.settings.num_trace_samples(10), \
                    gp.settings.preconditioner_tolerance(1e-3), \
                    gp.settings.tridiagonal_jitter(1e-5), \
                    gp.settings.cholesky_jitter(1e-3):

            print(' \n Making predictions for {0} model...'.format(name))
            start = time.time()
            if hasattr(models[name], 'full_likelihood'):  # we have to compute the full likelihood of projected models
                full_likelihood = models[name].full_likelihood()
            else:
                full_likelihood = likelihoods[name]
            ##------------------------------------------------------------------
            ## Computing, displaying and storing performance metrics

            observed_pred = full_likelihood(models[name](X_test))
            pred_y = observed_pred.mean
            if skip_var:
                var_pred = models[name].compute_var(X_test)
            else:
                var_pred = observed_pred.variance
            sigma_pred = var_pred.sqrt().squeeze()
            pred_time = time.time() - start

            global_noise = full_likelihood.noise.squeeze() if hasattr(full_likelihood, 'noise') else 0.
            if hasattr(full_likelihood, 'task_noise_covar_factor'):
                H_guess_hid = full_likelihood.task_noise_covar_factor.squeeze()
                H_guess_hid[range(n_tasks), range(n_tasks)] = H_guess_hid.diag() + global_noise
            elif hasattr(full_likelihood, 'task_noises'):
                H_guess_hid = (full_likelihood.task_noises.squeeze() + global_noise).sqrt()
            else:
                H_guess_hid = torch.ones(n_tasks) * global_noise.sqrt()

            metrics = compute_metrics(Y_test, pred_y, sigma_pred, last_losses[name], H_guess_hid, 
                                     effective_n_iters[name], times[name], pred_time, print_metrics=print_metrics, test_mask=test_mask)
            metrics.update(v)
            metrics['model'] = name
            results[name + run_key] = metrics
            preds_y[name] = pred_y
            sigma[name] = sigma_pred
    return results, models, preds_y, sigma

##----------------------------------------------
## Neutronics

## Generic setup 

print_metrics=True  # if True, performance metrics are printed at each run (dosen't affect exported results)
print_loss=True # if True, loss is printed after freq_print iteration (dosen't affect exported results)
freq_print=1000 # to further customize experiment name
train_ind_rat = 1.5 # ratio between number of training points and number of inducing points for the variational model
gpu = False
export_results = False
export_data_to_np = False
models_to_run = ['proj']
mods_to_train = []
# models_to_run = ['proj']
komi_format = True
file_format = 'pickle' # 'json' or 'pickle'
v_test_2 = 'void'
n_ind_points = None

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
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

    root = '_experiments/neutro_data/'
    X = torch.load(root + 'train_x_sobol256.pt')
    X_test = torch.load(root + 'test_x_LHS512.pt')
    Y = torch.load(root + 'train_data_02g_FA_Lchain.pt')
    Y_test = torch.load(root + 'test_data_02g_FA_Lchain.pt')
    means = torch.load(root + 'means_02g_FA_Lchain.pt')
    maxes = torch.load(root + 'maxes_02g_FA_Lchain.pt')
    # X, X_test, Y, Y_test = X[::2], X_test[::2], Y[::2, :30], Y_test[::2, :30]

    n_points, n_tasks = Y.shape
    case = 'b0py26e_02g_FA_meancrit'  # enter the name of the run here
    exproot = '//tmpcatB/KOMINTERN/TP/data/'
    if export_data_to_np:
        xs_keys = pickle.load(open(root + 'keys_02g_FA_Lchain_full.pickle', 'rb'))
        np.save(exproot + 'test_x_LHS512_full.npy', X_test.numpy())
        np.save(exproot + 'train_data_02g_FA_Lchain_full.npy', Y.numpy())
        np.save(exproot + 'test_data_02g_FA_Lchain_full.npy', Y_test.numpy())
    
    lr_min = 1e-3
    lr_max = 1e-1
    n_iter = 60000
    stopp_crit = 'mean'
    loss_thresh = 2e-5 if stopp_crit == 'max' else 1e-7
    patience = 50 if stopp_crit == 'max' else 500
    last_epoch = 15000
    lambda_f = lambda i : i/last_epoch*lr_min/lr_max + (last_epoch-i)/last_epoch if i <= last_epoch else lr_min/lr_max
    n_iters = dict(zip(models_to_run, [n_iter]*len(models_to_run)))
    lrs = dict(zip(models_to_run, [lr_max]*len(models_to_run)))
    models_with_sched = models_to_run
    ker_type = gp.kernels.RBFKernel # gp.kernels.PiecewisePolynomialKernel
    ker_type = SplineKernel
    mean_type = gp.means.ZeroMean
    decomp = [[0,1,2,3]]
    v = {
        'n_lat': 12, 
        'lik_rank': 0,
        'void' : [0.]}
    v_vals = {
        'n_lat' : range(1,n_tasks+1), 
        'lik_rank' : [0, n_tasks],
        'void' : [0.]}
    v_test = 'void'

    results = {}
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v[v_test] = vval
            v[v_test_2] = vval2
            n_lat, lik_rank = v['n_lat'], v['lik_rank']
            run_key = '_' + v_test + '_' + v_test_2 + '_{0}_{1}'.format(i_v, i_v2)
            results, models, preds_y, sigma = run_models(models_to_run, models_with_sched, n_lat, lik_rank, n_tasks, n_points, X, Y, X_test, Y_test, lrs, 
                    n_iters, lr_min, loss_thresh, patience, print_metrics, print_loss, freq_print, gpu, 
                    train_ind_rat, n_ind_points, run_key, results, decomp=decomp, lambda_lr=lambda_f, kernel_type=ker_type, mean_type=mean_type)
    df = pd.DataFrame.from_dict(results, orient='index')

    dico = {}
    run_info = {}
    dico['case'] = case
    dico['X'] = X.tolist()
    dico['y_means'] = means.tolist()
    dico['y_maxes'] = maxes.tolist()
    dico['y_labels'] = xs_keys.tolist()
    dico['x_labels'] = ['Bu', 'Tf', 'Tm', 'Cb']
    dico['x_bounds'] = [[0, 62000], [373.15, 2073.15], [373.15, 600], [0, 2500]]
    dico['kernel_decomp'] = decomp
    dico['kernel_type'] = ker_type.__name__
    for name in models_to_run:
        subdico, subinfo = {}, {}
        subinfo['pred_y'] = preds_y[name].tolist()
        subinfo['pred_sigma'] = sigma[name].tolist()
        subinfo['pred_time'] = df.loc[name + run_key, 'pred_time'].tolist()
        subinfo['pred_results'] = results[name + run_key]
        if komi_format:
            subdico['lscales'] = models[name].lscales().tolist()
            if name in ['ICM', 'var']:
                subdico['lmc_coeffs'] = models[name].lmc_coefficients().detach().tolist()
                noises = models[name].likelihood.task_noises.detach() + models[name].likelihood.noise.detach()
                subdico['noises'] = noises.tolist()
            else:
                subdico['Q'] = models[name].lmc_coefficients.Q().detach().tolist()
                subdico['R'] = models[name].lmc_coefficients.R.detach().tolist()
                subdico['Sigma_proj'] = models[name].projected_noise().detach().tolist()
                subdico['Sigma_orth'] = torch.exp(models[name].log_B_tilde.detach()[0]).tolist()

            if name=='proj':
                _ = models[name](X_test) # this is to compute the mean cache
                subdico['mean_cache'] = models[name].prediction_strategy.mean_cache.tolist()
            elif name=='var':
                with gp.settings.skip_posterior_variances(state=True):
                    _ = models['var'](X_test) # this is to compute the mean cache
                    alt_pred = models['var'](X_test).mean.squeeze().tolist() # this is to compute preds with the alternative formula
                    subinfo['pred_y'] = alt_pred
                subdico['mean_cache'] = models['var'].variational_strategy.base_variational_strategy._mean_cache.squeeze().detach().tolist()
                subdico['inducing_points'] = models['var'].variational_strategy.base_variational_strategy.inducing_points.detach().tolist()
                subdico['distrib_covar'] = models['var'].variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.detach().tolist()
            elif name=='ICM':
                with gp.beta_features.checkpoint_kernel(0),\
                        gp.settings.skip_posterior_variances(state=True):
                    pred = models['ICM'](X_test).mean.squeeze()
                    gp_cache = models[name].prediction_strategy.mean_cache
                    res = gp_cache.reshape((n_points, n_tasks)).matmul(models[name].covar_module.task_covar_module.covar_matrix)
                    subdico['mean_cache'] = res.detach().tolist()

        else:
            torch.save(models[name].state_dict(), exproot + 'model_{0}.pth'.format(name))  #standard pytorch format
            dico['Y'] = Y.tolist()
            if name == 'var':
                torch.save(models[name].likelihood.state_dict(), exproot + 'likelihood_{0}.pth'.format(name))

        dico[name + '_' + 'params'] = subdico
        run_info[name] = subinfo

    if file_format == 'pickle':
        if len(models_to_run) == 1:
            case += '_' + models_to_run[0] + '_test'
        with open(exproot + 'gp_{0}_run_info.pkl'.format(case), 'wb') as f:
            pickle.dump(run_info, f)
        with open(exproot + 'gp_data_{0}.komi'.format(case), 'wb') as f:
            pickle.dump(dico, f)
    elif file_format == 'json':
        with zipfile.ZipFile(exproot + '{0}.zip'.format(case), 'w', zipfile.ZIP_DEFLATED) as zf:
        # Écrire le dictionnaire au format JSON dans un fichier à l'intérieur de l'archive
            zf.writestr(exproot + 'gp_data_{0}.komi.json'.format(case), json.dumps(dico, indent = 3))
        with zipfile.ZipFile(exproot + '{0}.zip'.format(case), 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(exproot + 'gp_{0}_run_info.json'.format(case), json.dumps(dico, indent = 3))




