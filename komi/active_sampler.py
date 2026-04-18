import numpy as np
import torch
import gpytorch as gp
import pandas as pd
from scipy.stats import qmc
from scipy.spatial.distance import cdist

from komi.mogp_plmc import ProjectedGPModel
from komi.mogp_lazy import LazyLMCModel

## Utilitaires
from utilities import CorrectReduceLROnPlateau
def get_closest_el(l, elem):
    # on suppose que la liste est triée et qu'on sera toujours au-dessus du plus petit élément
    previous_item = None
    for item in l:
        if elem < item :
            return previous_item
        previous_item = item
    return previous_item

def prod_func(x, dim=None):
    return torch.sum(torch.log(x), dim=dim)

def max_func(x, dim=None):
    return torch.max(x, dim=dim).values

def sum_func(x, dim=None):
    return torch.sum(x, dim=dim)

##----------------------------------------------------------------------------------------------------------------------
class ActiveSampler:

    def __init__(self, model, strategy, aggr_func, current_data=None, current_X=None, candidate_X=None, **kwargs):
        
        self.model = model
        self.strategy = strategy
        self.aggr_func = aggr_func
        self.current_data = current_data
        self.current_X = current_X
        self.visited_points = []
        # Initialize candidate set if provided
        if candidate_X is not None:
            self.X_candidates = candidate_X.to(torch.get_default_dtype())
            self.old_indices = np.arange(len(self.X_candidates))
        elif self.strategy in ["Tdownsampling","Ldownsampling"]:
            self.old_indices = np.arange(len(self.current_X))
        # Normalize initial training data and store statistics
        if self.current_data is not None:
            mean = self.current_data.mean(dim=0)
            std = torch.std(self.current_data, dim=0)
            self.current_data = (self.current_data - mean) / std
            self.train_mean = mean
            self.train_std = std

    def gen_candidate_set(self, n_points, dim, algo='sobol', seed=0, return_set=False):
        if algo == 'sobol':
            sobol = qmc.Sobol(d=dim, seed=seed)
            m = int(np.ceil(np.log2(n_points)))
            set = 2*sobol.random_base2(m=m) - 1
        elif algo == 'LHS':
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
            set = 2*sampler.random(n=n_points) - 1
        elif algo == 'random':
            set = 2*np.random.rand(n_points, dim) - 1
        else:
            raise NotImplementedError
        self.X_candidates = torch.tensor(set).to(torch.get_default_dtype())
        self.old_indices = np.arange(len(self.X_candidates))
        if return_set:
            return set
        
    def add_data(self, X, Y, normalize=False, norm_func=torch.std):
        self.current_X = torch.cat([self.current_X, X])
        # De-normalize existing data if statistics exist
        if hasattr(self, 'train_mean') and self.train_mean is not None:
            existing = self.current_data * self.train_std + self.train_mean
        else:
            existing = self.current_data
        # Stack with new raw data
        combined = torch.cat([existing, Y])
        # Compute new statistics
        mean = combined.mean(dim=0)
        std = torch.std(combined, dim=0)
        # Normalize combined data
        self.current_data = (combined - mean) / std
        # Update stored statistics
        self.train_mean = mean
        self.train_std = std

    def modify_train_set(self, new_X=None, new_Y=None, normalize=False, norm_func=torch.std):
        if new_X is not None and new_Y is not None:
            self.add_data(new_X, new_Y, normalize=normalize)
        train_x, train_y = self.current_X, self.current_data
        if self.strategy in ["Tdownsampling","Ldownsampling"]:
            mask = np.array([point not in self.visited_points for point in np.arange(len(train_x))]).astype(bool)
            train_x, train_y, self.old_indices = train_x[mask], train_y[mask], np.arange(len(train_x))[mask]
        else:
            mask = np.array([point not in self.visited_points for point in np.arange(len(self.X_candidates))]).astype(bool)
            self.X_candidates = self.X_candidates[mask]
            self.old_indices = np.arange(len(self.X_candidates))[mask]
        
        self.model.set_train_data(train_x, train_y, strict=False)

        # if self.strategy!='downsampling':
        #     new_x, new_y = train_x[-self.n_samples:], train_y[-self.n_samples:]
        #     if self.n_samples==1:
        #         new_x, new_y = new_x.unsqueeze(0), new_y.unsqueeze(0)
        #     self.model = self.model.get_fantasy_model(new_x, new_y)

    def train_model(self, training_settings):
        """Train the sampler's model using the exact stopping logic from realdata_experiments.
        training_settings: (lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched)
        Returns (best_loss, effective_iters).
        """
        lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched = training_settings
        # Setup model and optimizer
        self.model.train()
        likelihood = self.model.likelihood
        likelihood.train()
        mll = self.model.default_mll()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_max)
        scheduler = CorrectReduceLROnPlateau(optimizer, factor=0.5, patience=patience_sched,
                                            threshold=loss_thresh, threshold_mode='rel', mode='min', min_lr=lr_min)
        best_loss = 1e9
        last_lr = 1.
        no_improve_count = 0
        effective_iters = n_iters
        for i in range(n_iters):
            optimizer.zero_grad()
            with gp.settings.cholesky_max_tries(8):
                output = self.model(self.current_X)
                loss = -mll(output, self.current_data).squeeze()
                loss.backward()
                optimizer.step()
                new_loss = loss.item()
            # Scheduler step
            scheduler.step(new_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < last_lr:
                last_lr = current_lr
                no_improve_count = 0
            # Improvement checks
            loss_better_than_best = (new_loss >= 0. and new_loss < best_loss * (1 - loss_thresh)) \
                                    or (new_loss < 0. and new_loss < best_loss * (1 + loss_thresh))
            loss_almost_as_good_as_best = (new_loss >= 0. and new_loss < best_loss * (1 + loss_thresh)) \
                                    or (new_loss < 0. and new_loss < best_loss * (1 - loss_thresh))
            if loss_better_than_best:
                best_loss = new_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            if use_stop and (no_improve_count > patience_crit) and loss_almost_as_good_as_best:
                effective_iters = i
                break
        return best_loss, effective_iters

    def compute_scores(self, var_values, lscales_mat, e_loo2, s_loo2, lmc_coeffs, aggregated=True):
        space_size = lscales_mat.shape[0]   
        weights = torch.zeros((len(self.X_candidates), space_size))
        for i in range(space_size):
            dist_mat = cdist(self.X_candidates.cpu().numpy() / lscales_mat[i,:].cpu().numpy(), self.current_X.values / lscales_mat[i,:].cpu().numpy())
            nn_index = np.argmin(dist_mat, axis=1)
            weights[:,i] = e_loo2[nn_index, i] / s_loo2[nn_index, i]
        if var_values.is_cuda:
          weights = weights.cuda()
        objective_function = var_values * (1 + weights)
        if self.strategy=='Tloo':
            mags = torch.linalg.norm(lmc_coeffs, dim=1)
            scores = (objective_function * mags[None,:])
        else:
            scores = objective_function
        if aggregated:
            return self.aggr_func(scores, dim=1)
        else:
            return scores

    def find_next_points(self, n_samples, output_train=None, only_idx=False, verbose=True, **kwargs):
        if self.strategy in ['Lloo', 'Tloo', 'Ldownsampling', 'Tdownsampling']:
            var_loo, delta_loo = self.model.compute_loo(output_train, latent=(self.strategy in ['Lloo', 'Ldownsampling']))
            if self.strategy in ['Lloo', 'Tloo']:
                eloo2 = delta_loo**2
            else:
                eloo = torch.abs(delta_loo)

        # assumes no kernel decomposition. Otherwise, need to modify this (with some heuristic aggregation of lengthscales)
        if self.strategy in ["Ldownsampling","Tdownsampling"]:
            scores = self.aggr_func(eloo, dim=1)
        else:
            if self.strategy in ['Tloo', 'Lloo', 'Lvar']:
                output = self.model.likelihood(self.model.compute_latent_distrib(self.X_candidates))
                var_values = output.variance
                var_values = var_values.T
            else:
                full_lik = self.model.full_likelihood()
                output = full_lik(self.model(self.X_candidates))
                var_values = output.variance

            if self.strategy in ['Tvar', 'Lvar']:
                scores = self.aggr_func(var_values, dim=1)
            else:
                lscales_mat = self.model.lscales()
                lmc_coeffs = self.model.lmc_coefficients() if self.strategy=='Tloo' else None
                scores = self.compute_scores(var_values, lscales_mat, eloo2, var_loo, lmc_coeffs)

        if self.strategy in ["Ldownsampling","Tdownsampling"]:
            top_values, top_ind_red = torch.topk(torch.as_tensor(scores), n_samples, largest=False)
            top_ind = self.old_indices[top_ind_red]
            new_points = top_ind if only_idx else self.current_X[top_ind]
        else:
            self.metrics['score'] = scores.max().cpu().numpy()
            top_values, top_ind_red = torch.topk(torch.as_tensor(scores), n_samples)
            top_ind = self.old_indices[top_ind_red]
            new_points = top_ind if only_idx else self.X_candidates[top_ind].numpy()
        if verbose:
            best_score = top_values[0].cpu().numpy()
            print('Score at current iteration :', best_score)

        if n_samples==1:
            new_points = np.expand_dims(new_points,0) if not only_idx else new_points
            self.visited_points.append(top_ind.item())
        else:
            for ind in top_ind:
                self.visited_points.append(ind.item())

        return new_points, best_score


        
        
        
