import time
import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from komi.utilities import tensor_iter, compute_macro_errs

class BayesPolyRegressor :

    def __init__(self, n_tasks, degree, tol, max_iter, model = BayesianRidge, **kwargs):
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.models = [make_pipeline(PolynomialFeatures(degree), model(tol=tol, compute_score=False, max_iter=max_iter, **kwargs))
                          for _ in range(n_tasks)]

    def train( self, train_x, train_y, compute_score=True, coeff_info=False):
        n_funcs = train_y.shape[1]
        n_coeffs = np.zeros(n_funcs)
        
        start = time.time()
        loss = 0.
        for i in range(train_y.shape[1]):
            model = self.models[i]
            if compute_score:
                model.steps[1][1].compute_score = True
            model.fit(train_x, train_y[:,i])

            if i == 0:
                # This block is only executed once, to get the monomials and the number of coefficients
                features = model.steps[0][1]
                monomials = features.get_feature_names_out() # The model must be fitted for this to be defined. 
                coeff_list = np.zeros((n_funcs, len(monomials)))

            reg = model.steps[1][1] # reg is the regression part of the model (not the polynomial features)
            n_coeffs[i] = len(reg.coef_[reg.coef_ != 0.])
            if coeff_info:
                coeff_list[i] = np.abs(reg.coef_)
            if compute_score:
                loss += reg.scores_[-1]
                model.steps[1][1].compute_score = False # Resetting the compute_score flag to default

        end = time.time()
        loss /= n_funcs

        stats = {'n_coeffs_av': n_coeffs.mean(), 'duration': end-start, 'loss': -loss}
        if coeff_info:
            features, reg = self.model.steps[0][1], self.model.steps[1][1]
            mean_coeffs = coeff_list.mean(axis=0)
            idx = np.argsort(mean_coeffs)[::-1]
            monomials = features.get_feature_names_out()
            coeff_info = dict(zip(monomials[idx], mean_coeffs[idx]))    
            return stats, coeff_info
        else:
            return stats


    def predict( self, test_x, return_std=False):
        n_points = test_x.shape[0]
        n_funcs = len(self.models)
        y_pred = np.zeros((n_points, n_funcs))
        y_pred_std = np.zeros((n_points, n_funcs))
        for i in range(n_funcs):
            y_pred[:,i], y_pred_std[:,i] = self.models[i].predict(test_x, return_std=return_std)
        return y_pred, y_pred_std
    
    def evaluate( self, test_x, test_y, met_dict={}, concs=None, keys=None, devs=None):
        n_points, n_funcs = test_y.shape
        raw_met_tags = ['y_test', 'deltas', 'errs', 'errs2', 'sigmas', 'vars', 'keys', 'concs']
        raw_metrics = {tag: np.zeros((n_points, n_funcs)) for tag in raw_met_tags}
        raw_metrics['y_test'] = test_y
        raw_metrics['keys'] = keys
        raw_metrics['concs'] = concs

        start = time.time()
        y_pred, raw_metrics['sigmas'] = self.predict(test_x, return_std=True)
        end = time.time()

        stats = {'duration': end-start}
        raw_metrics['deltas'] = test_y - y_pred
        raw_metrics['errs'] = np.abs(raw_metrics['deltas'])
        raw_metrics['errs2'] = raw_metrics['errs']**2
        raw_metrics['vars'] = raw_metrics['sigmas']**2
        if devs is not None:
            raw_metrics['u_errs'] = devs * raw_metrics['errs']

        if met_dict == {}:
            raw_metrics.update(stats)
            return raw_metrics
        else:
            for met in met_dict:
                stats[met] = met_dict[met](raw_metrics)
            return stats
