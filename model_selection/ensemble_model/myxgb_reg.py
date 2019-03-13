#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import xgboost as xgb
import gc
from sklearn.externals import joblib

class Myxgb_regression(object):
    """myxgb is best tool for data mining"""
    def __init__(self, config):
        if "params" in config:
            self.params = config["params"]
        else:
            self.set_default_params()
        if "xgb_data" in config:
            self.tr_data = config["xgb_data"]
        else:
            self.tr_data = None
        if "xgb_label" in config:
            self.tr_label = config["xgb_label"]
        else:
            self.tr_label = None
        if "nfold" in config:
            self.nfold = config["nfold"]
        else:
            self.nfold = 5
        if "num_boost_round" in config:
            self.num_boost_round = config['num_boost_round']
        else:
            self.num_boost_round = 5000
        if "seed" in config:
            self.seed = config["seed"]
        else:
            self.seed = 666
        if "early_stopping_rounds" in config:
            self.early_stopping_rounds = config["early_stopping_rounds"]
        else:
            self.early_stopping_rounds = 100
        if "verbose_eval" in config:
            self.verbose_eval = config[verbose_eval]
        else:
            self.verbose_eval = 100
        self.min_merror = float('Inf')
        self.model = None

    def get_best_thres(self):
        pass

    def set_tr_data(self, data):
        self.tr_data = data

    def train(self):
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        cv_results = xgb.cv(
                            params = self.params,
                            dtrain = xgb_train,
                            seed = self.seed,
                            nfold = self.nfold,
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = self.early_stopping_rounds,
                            verbose_eval = self.verbose_eval
                           )
        boost_rounds = pd.Series(cv_results['test-mae-mean']).idxmin()
        self.score = pd.Series(cv_results['test-mae-mean']).min()
        self.model = xgb.train(self.params, xgb_train, num_boost_round = boost_rounds)
        return True

    def set_default_params(self):
        self.params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric':'mae',
            'eta': 0.01,
            'subsample': 0.9,
            'seed': 77,
            'nthread': -1,
            'max_depth': 6,
            'verbose': 0,
            'silent': 1
            }


    def find_best_params(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        self.adj_gamma(seed, nfold, early_stopping_rounds)
        print ('has find best gamma:%s' % self.params['gamma'])
        self.adj_depth(seed, nfold, early_stopping_rounds)
        print ('has find best max_depth:%s' % self.params['max_depth'])
        self.adj_min_child_weight(seed, nfold, early_stopping_rounds)
        print ('has find best min_child_weight:%s' % self.params['min_child_weight'])
        self.adj_fraction(seed, nfold, early_stopping_rounds)
        print ('has find best colsample_bytree/subsample:%s/%s' % (self.params['colsample_bytree'], self.params['subsample']))
        self.adj_lambda(seed, nfold, early_stopping_rounds)
        print ('has find best alpha/lambda:%s/%s' % (self.params['alpha'], self.params['lambda']))
        return True

    def adj_eta(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        for eta in [0.01, 0.015, 0.025, 0.05, 0.1]:
            self.params['eta'] = eta
            cv_results = xgb.cv(
                                self.params,
                                xgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['test-mae-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['eta'] = eta
        self.params.update(best_params)

    def adj_gamma(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        for gamma in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            self.params['gamma'] = gamma
            cv_results = xgb.cv(
                                self.params,
                                xgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['test-mae-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['gamma'] = gamma
        self.params.update(best_params)

    def adj_depth(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        for max_depth in [3,5,7,9,12,15,17,25]:
            self.params['max_depth'] = max_depth
            cv_results = xgb.cv(
                                self.params,
                                xgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['test-mae-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['max_depth'] = max_depth
        self.params.update(best_params)

    def adj_min_child_weight(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        for min_child_weight in [1, 3, 5, 7]:
            self.params['min_child_weight'] = min_child_weight
            cv_results = xgb.cv(
                                self.params,
                                xgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['test-mae-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['min_child_weight'] = min_child_weight
        self.params.update(best_params)

    def adj_fraction(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1]:
            for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1]:
                self.params['colsample_bytree'] = feature_fraction
                self.params['subsample'] = bagging_fraction
                cv_results = xgb.cv(
                                    self.params,
                                    xgb_train,
                                    seed = seed,
                                    nfold = nfold,
                                    num_boost_round = self.num_boost_round,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                mean_merror = pd.Series(cv_results['test-mae-mean']).min()

                if mean_merror <= self.min_merror:
                    self.min_merror = mean_merror
                    best_params['colsample_bytree'] = feature_fraction
                    best_params['subsample'] = bagging_fraction
        self.params.update(best_params)

    def adj_lambda(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        xgb_train = xgb.DMatrix(self.tr_data, self.tr_label)
        for lambda_l1 in [0.0, 0.1, 0.5, 1.0]:
            for lambda_l2 in [0.0, 0.01, 0.1, 1.0]:
                self.params['alpha'] = lambda_l1
                self.params['lambda'] = lambda_l2
                cv_results = xgb.cv(
                                    self.params,
                                    xgb_train,
                                    seed = seed,
                                    nfold = nfold,
                                    num_boost_round = self.num_boost_round,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
            mean_merror = pd.Series(cv_results['test-mae-mean']).min()
            if mean_merror <= self.min_merror:
                self.min_merror = mean_merror
                best_params['alpha'] = lambda_l1
                best_params['lambda'] = lambda_l2
        self.params.update(best_params)

    # TODO: LGB粒子群调参法
    def pso_find_best_params(self):
        pass

    def save_model(self, model_path):
        joblib.dump(self.model,model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
