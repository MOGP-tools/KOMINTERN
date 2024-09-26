import time
from contextlib import ExitStack
import concurrent.futures
import warnings
import tqdm

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
import gpytorch as gp

from .base_gp import ExactGPModel
from .mogp_lazy import LazyLMCModel
from .mogp_var import VariationalMultitaskGPModel
from .mogp_plmc import ProjectedGPModel, ProjectedLMCmll
from .mogp_icm import MultitaskGPModel

train_context_managers = {
    'skldf':gp.settings.skip_logdet_forward(state=False),
    'cgtol':gp.settings.cg_tolerance(1e-0),
    'ecgtol':gp.settings.eval_cg_tolerance(1e-2),
    'maxchsize':gp.settings.max_cholesky_size(128),
    'maxlqi':gp.settings.max_lanczos_quadrature_iterations(20),
    'maxprecsize':gp.settings.max_preconditioner_size(15),
    'maxrootsize':gp.settings.max_root_decomposition_size(100),
    'minprecsize':gp.settings.min_preconditioning_size(2000),
    'ntracesamp':gp.settings.num_trace_samples(10),
    'precondtol':gp.settings.preconditioner_tolerance(1e-3),
    'tridjitter':gp.settings.tridiagonal_jitter(1e-5),
    # 'maxcgiter':gp.settings.max_cg_iterations(5000),
}
test_context_managers = {
    'nograd':torch.no_grad(),
    'skldf':gp.settings.skip_logdet_forward(state=False),
    'cgtol':gp.settings.cg_tolerance(1e-0),
    'ecgtol':gp.settings.eval_cg_tolerance(1e-2),
    'maxchsize':gp.settings.max_cholesky_size(128),
    'maxlqi':gp.settings.max_lanczos_quadrature_iterations(20),
    'maxprecsize':gp.settings.max_preconditioner_size(15),
    'maxrootsize':gp.settings.max_root_decomposition_size(100),
    'minprecsize':gp.settings.min_preconditioning_size(2000),
    'ntracesamp':gp.settings.num_trace_samples(10),
    'precondtol':gp.settings.preconditioner_tolerance(1e-3),
    'tridjitter':gp.settings.tridiagonal_jitter(1e-5),
}

def load_model(model): #, X=None, Y=None):
    if isinstance(model, tuple):
        ## warning : Passing a model dictionary to train_model is only tested for SOGPS. Any other model will likely result in an error.
        mtype, margs, mstate = model
        # margs = margs.copy()
        # if 'train_x' not in margs:
        #     margs['train_x'] = X
        # if 'train_y' not in margs:
        #     margs['train_y'] = Y
        # print(margs)
        model = mtype(**margs)
        model.load_state_dict(mstate, strict=False)
    return model

def train_model(model_init, X, Y, argus, optimizer=None, compute_loo=False, return_optim=False, met_dict={}, extra_context_managers={}, 
                keys=None, concs=None, devs=None, verbose=True):
    # try:
    if len(Y.shape) == 1:
        n_points, n_tasks = Y.shape[0], 1
    else:
        n_points, n_tasks = Y.shape
    model = load_model(model_init) #, X=X, Y=Y)
    if 'minibatch_size' in argus and argus['minibatch_size'] is not None:
        minibatch_size = argus['minibatch_size']
        train_dataset = torch.utils.data.TensorDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True, 
                                                    num_workers=0, pin_memory=argus['gpu'])
    else:
        minibatch_size = None

    if argus['gpu']:
        X = X.cuda()
        Y = Y.cuda()
        model = model.cuda()

    model.train()
    mll = model.default_mll()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=argus['lr_max'])

    if 'lambda_lr' in argus:
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, argus['lambda_lr'])
    else:
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(argus['lr_min'] / argus['lr_max']) / argus['n_iter']))

    n_it_eff = argus['n_iter']
    if verbose:
        print(' \n Entering training ... \n')
    start = time.time()
    plateau_id = 0
    deltas = np.zeros(argus['patience'])
    context_managers = train_context_managers.copy()
    context_managers.update(extra_context_managers)
    with ExitStack() as stack:
        for _, context in context_managers.items():
            stack.enter_context(context)

        for i in range(argus['n_iter']):
            optimizer.zero_grad()

            new_loss = 0.
            if minibatch_size is not None:
                loss = None
                for x_batch, y_batch in train_loader: # The whole dataset is loaded at each iteration. Other choices are possible.
                    if argus['gpu']:
                        x_batch, y_batch = x_batch.cuda(non_blocking=False), y_batch.cuda(non_blocking=False)
                    if loss is not None: # tester l'effet de ceci
                        del loss
                    optimizer.zero_grad()
                    output_batch = model(x_batch)
                    loss = -mll(output_batch, y_batch)
            else:
                output_train = model(X)
                loss = -mll(output_train, Y)
                
            new_loss += loss.item()
            if not isinstance(model, MultitaskGPModel):
                new_loss /= n_tasks
            if verbose and argus['print_loss'] and i%argus['freq_print']==0:
                print(new_loss)
            loss.backward()
            optimizer.step()
            sched.step()

            if argus['stopp_crit'] == 'max':
                if i>0 and (np.abs(last_loss) < 1e-5 or np.abs( 1 - new_loss / last_loss) < argus['loss_thresh']):
                    plateau_id += 1
                    if plateau_id > argus['patience'] :
                        n_it_eff = i
                        break
                else:
                    plateau_id = 0
            elif argus['stopp_crit'] == 'mean':
                if i > 0:
                    for j in range(argus['patience'] - 1):
                        deltas[j+1] = deltas[j]
                    deltas[0] = 0 if np.abs(last_loss) < 1e-5 else np.abs( 1 - new_loss / last_loss)
                if i >= argus['patience'] and deltas.mean() < argus['loss_thresh']:
                    n_it_eff = i
                    break
            elif argus['stopp_crit'] == 'exp':
                if i > 0 and np.abs(1 - np.exp(new_loss - last_loss)) < argus['loss_thresh']:
                        n_it_eff = i
                        break
            else:
                raise ValueError('Unknown stopping criterion')
            last_loss = new_loss

    new_time = time.time()
    duration = new_time - start
    stats = {'loss': new_loss, 'n_iter': n_it_eff, 'train_duration': duration}

    if compute_loo:
        with torch.no_grad():
            loo_vars, loo_deltas = model.compute_loo(output=output_train)
            loo_duration = time.time() - new_time
            stats['loo_duration'] = loo_duration
            if not met_dict: # if no metrics are specified, we return the raw data
                stats['loo_vars'] = loo_vars.cpu().numpy()
                stats['loo_deltas'] = loo_deltas.cpu().numpy()
            else:
                raw_metrics = {'y_test':Y, 'deltas':loo_deltas, 'errs':torch.abs(loo_deltas), 'errs2':loo_deltas**2,
                            'sigmas':torch.sqrt(loo_vars), 'vars':loo_vars, 'keys':keys, 'concs':concs}
                if devs is not None:
                    raw_metrics['u_errs'] = devs * raw_metrics['errs']
                for met in met_dict:
                    stats[met] = met_dict[met](raw_metrics).cpu().detach().numpy()
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    if isinstance(model_init, tuple):
        model = (model_init[0], model_init[1], model.state_dict())
    if return_optim:
        return model, stats, optimizer
    return model, stats

def predict(model, X_test, gpu=False, extra_context_managers={}, return_full_lik=False):
    context_managers = test_context_managers.copy()
    context_managers.update(extra_context_managers)
    with ExitStack() as stack:
        for _, context in context_managers.items():
            stack.enter_context(context)
        if gpu:
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()
            model = model.cuda()
            if hasattr(model, 'full_likelihood'):  # we have to compute the full likelihood of projected models
                full_likelihood = model.full_likelihood()
            else:
                full_likelihood = model.likelihood
            free_mem = torch.cuda.mem_get_info()[0]
            num_bytes = X_test.element_size()
            n_points, n_tasks = model.train_targets.shape
            batch_size = int(free_mem / (16 * n_points * n_tasks * num_bytes)) # à optimiser proprement
            preds, vars = [], [] 
            for i in range(0, len(X_test), batch_size):
                x_batch = X_test[i:i+batch_size]
                observed_pred = full_likelihood(model(x_batch))
                pred_y = observed_pred.mean
                preds.append(pred_y)
                ## the default variance computaton of the ICM is very memory-intensive. We replace it with our custom formula.
                ## The function compute_var already operates on batches, so we call it outside the loop.
                if not (context_managers.get('skip_pred_var', True) and isinstance(model, MultitaskGPModel)): 
                    vars_pred = observed_pred.variance
                    vars.append(vars_pred)
            pred_y = torch.cat(preds)
            if context_managers.get('skip_pred_var', True) and isinstance(model, MultitaskGPModel):
                vars_pred = model.compute_var(X_test)
            else:
                vars_pred = torch.cat(vars)
        else:
            model = model.cpu()
            observed_pred = full_likelihood(model(X_test))
            pred_y = observed_pred.mean
            if context_managers.get('skip_pred_var', True) and isinstance(model, MultitaskGPModel):
                vars_pred = model.compute_var(X_test)
            else:
                vars_pred = observed_pred.variance
    
    if return_full_lik :
        return pred_y, vars_pred, full_likelihood
    return pred_y, vars_pred


def eval_model(model, X_test, Y_test, argus, met_dict, extra_context_managers={}, concs=None, devs=None, keys=None, verbose=True):
    model = load_model(model)
    model.eval()
    start = time.time()
    if verbose:
        print(' \n Making predictions...')

    pred_y, vars_pred, full_likelihood = predict(model, X_test, gpu=argus['gpu'], extra_context_managers=extra_context_managers,
                                                 return_full_lik=True)
    pred_time = time.time() - start

    ## Computation of some noise terms
    global_noise = full_likelihood.noise.squeeze() if hasattr(full_likelihood, 'noise') else 0.
    n_tasks = Y_test.shape[1]
    if hasattr(full_likelihood, 'task_noise_covar_factor'):
        noise_mat_root = full_likelihood.task_noise_covar_factor.squeeze()
        noise_mat = noise_mat_root.matmul(noise_mat_root.t()) + global_noise * torch.eye(n_tasks)
        av_noise = torch.linalg.norm(torch.diag(noise_mat)) / n_tasks
    elif hasattr(full_likelihood, 'task_noises'):
        av_noise = torch.linalg.norm(full_likelihood.task_noises.squeeze() + global_noise)
    else:
        av_noise = global_noise

    metrics = {'pred_time': pred_time, 'noise': av_noise.cpu().numpy()}
    deltas = Y_test - pred_y
    sigmas_pred = vars_pred.sqrt()
    raw_metrics = {'y_test':Y_test, 'deltas':deltas, 'errs':torch.abs(deltas), 'errs2':deltas**2, 
                   'sigmas':sigmas_pred, 'vars':vars_pred, 'concs':concs, 'keys':keys}
    if devs is not None:
        raw_metrics['u_errs'] = devs * raw_metrics['errs']
    for met in met_dict:
        metrics[met] = met_dict[met](raw_metrics).cpu().numpy()
    return pred_y, sigmas_pred, metrics


def eval_loo(model, Y_train, keys=None, concs=None, devs=None, met_dict={}, extra_context_managers={}):
    model.train()
    start = time.time()
    context_managers = train_context_managers.copy()
    context_managers.update(extra_context_managers)
    ## Careful with the context ! cg_tolerance is different by default in training and testing, which has major consequences on the results.
    with ExitStack() as stack:
        for _, context in context_managers.items():
            stack.enter_context(context)
        print(' \n Computing leave-one-out errors...')
        loo_vars, loo_deltas = model.compute_loo()

    loo_duration = time.time() - start
    stats = {'loo_duration':loo_duration}
    raw_metrics = {'y_test':Y_train, 'deltas':loo_deltas, 'errs':torch.abs(loo_deltas), 'errs2':loo_deltas**2,
                    'sigmas':torch.sqrt(loo_vars), 'vars':loo_vars, 'keys':keys, 'concs':concs}
    if devs is not None:
        raw_metrics['u_errs'] = devs * raw_metrics['errs']
    for met in met_dict:
        stats[met] = met_dict[met](raw_metrics).cpu().numpy()
    return stats


def train_parallel(model_list, X, Y, argus, max_workers, compute_loo=False, keys=None, concs=None, devs=None, met_dict={}, 
                   extra_context_managers={}, aggregate=True):
    # Models must be stored in a dictionary to be pickable !
    n_points, n_tasks = Y.shape
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_model, model_list[i], X, Y[:,i], argus, 
                        compute_loo=compute_loo, extra_context_managers=extra_context_managers, verbose=False)
                    for i in range(n_tasks)]
        
    count = 0
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass # only to display a progression bar
        count += 1
        print(count)

    output_models = [None] * n_tasks
    losses, n_iters, durations = np.zeros(n_tasks), np.zeros(n_tasks), np.zeros(n_tasks)
    if compute_loo:
        loo_deltas = np.zeros_like(Y)
        loo_vars = np.zeros_like(Y)
        loo_dur = np.zeros(n_tasks)
    for i, future in enumerate(futures):
        model, stats = future.result()
        output_models[i] = model
        losses[i] = stats['loss']
        n_iters[i] = stats['n_iter']
        durations[i] = stats['duration']
        if compute_loo:
            loo_vars[:,i] = stats['loo_vars']
            loo_deltas[:,i] = stats['loo_deltas']
            loo_dur[i] = stats['loo_duration']

    if aggregate:
        stats = {'loss':losses.mean(), 'n_iter':n_iters.mean(), 'duration':durations.mean()}
    else:
        stats = {'loss':losses, 'n_iter':n_iters, 'duration':durations}

    if compute_loo:
        if not met_dict: # if no metrics are specified, we return the raw data
            stats['loo_vars'] = loo_vars.cpu().numpy()
            stats['loo_deltas'] = loo_deltas.cpu().numpy()
        else:
            raw_metrics = {'y_test':Y, 'deltas':loo_deltas, 'errs':torch.abs(loo_deltas), 'errs2':loo_deltas**2,
                        'sigmas':torch.sqrt(loo_vars), 'vars':loo_vars, 'keys':keys, 'concs':concs}
            if devs is not None:
                raw_metrics['u_errs'] = devs * raw_metrics['errs']
            for met in met_dict:
                stats[met] = met_dict[met](raw_metrics).cpu().numpy()

    return output_models, stats


def eval_parallel(model_list, X_test, Y_test, argus, max_workers, keys=None, concs=None, devs=None, met_dict={}, extra_context_managers={}):
    n_tasks = Y_test.shape[1]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(eval_model, model_list[i], X_test, Y_test[:,i], 
                                   argus, met_dict, extra_context_managers, verbose=False)
                    for i in range(n_tasks)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass # only to display a progression bar

    pred_ys, sigma_preds = np.zeros_like(Y_test), np.zeros_like(Y_test)
    pred_times, noises = np.zeros(n_tasks), np.zeros(n_tasks)
    for i, future in enumerate(futures):
        pred_y, sigma_pred, mets = future.result()
        pred_ys[:,i] = pred_y
        sigma_preds[:,i] = sigma_pred
        pred_times[i] = mets['pred_time']
        noises[i] = mets['noise']

    metrics = {'pred_time': pred_times.mean(), 'noise': noises.mean()}
    deltas = Y_test - pred_ys
    raw_metrics = {'y_test':Y_test, 'deltas':deltas, 'errs':torch.abs(deltas), 
                   'errs2':deltas**2, 'sigmas':sigma_preds, 'vars':sigma_preds**2, 'concs':concs, 'keys':keys}
    if devs is not None:
        raw_metrics['u_errs'] = devs * raw_metrics['errs']
    for met in met_dict:
        metrics[met] = met_dict[met](raw_metrics).cpu().numpy()

    return pred_ys, sigma_preds, metrics






