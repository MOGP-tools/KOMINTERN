import time
import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from komi.utilities import tensor_iter, compute_macro_errs

class BayesPolyRegressor :

    def __init__(self, n_tasks, degree, tol, n_iter, model = BayesianRidge, **kwargs):
        self.degree = degree
        self.tol = tol
        self.n_iter = n_iter
        self.models = [make_pipeline(PolynomialFeatures(degree), model(tol=tol, compute_score=True, n_iter=n_iter, **kwargs))
                          for _ in range(n_tasks)]

    def train( self, train_x, train_y, compute_score=True, coeff_info=False):
        self.fit_(train_x, train_y)
        n_funcs = train_y.shape[1]
        n_coeffs = np.zeros(n_funcs)
        
        features = self.models[0].steps[0][1]
        monomials = features.get_feature_names_out()
        coeff_list = np.zeros((n_funcs, len(monomials)))
        start = time.time()
        loss = 0.
        for i in range(train_y.shape[1]):
            model = self.models[i]
            model.fit(train_x, train_y[:,i], compute_score=compute_score)
            n_coeffs[i] = len(model.coef_[model.coef_ != 0.])
            reg = self.model.steps[1][1]
            if coeff_info:
                coeff_list[i] = np.abs(reg.coef_)
            if compute_score:
                loss += reg.scores_[-1]
        end = time.time()

        if coeff_info:
            features, reg = self.model.steps[0][1], self.model.steps[1][1]
            mean_coeffs = coeff_list.mean(axis=0)
            idx = np.argsort(mean_coeffs)[::-1]
            monomials = features.get_feature_names_out()
            coeff_info = dict(zip(monomials[idx], mean_coeffs[idx]))
        else:
            coeff_info = {}

        return {'n_coeffs_av': n_coeffs.mean(), 'coeff_info': coeff_info, 'duration': end-start, 'loss': loss}

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
