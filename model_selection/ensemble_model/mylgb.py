#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import lightgbm as lgb
import gc
from sklearn.metrics import f1_score
from sklearn.externals import joblib

class Mylgb(object):
    """mylgb is best tool for data mining"""
    def __init__(self, config):
        if "params" in config:
            self.params = config["params"]
        else:
            self.set_default_params()
        if "lgb_data" in config:
            self.tr_data = config["lgb_data"]
        else:
            self.tr_data = None
        if "lgb_label" in config:
            self.tr_label = config["lgb_label"]
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

        if self.params['metric'] in ['auc']:
            self.flag = -1
            self.min_merror = float('Inf') * -1
        else:
            self.flag = 1
            self.min_merror = float('Inf')
        self.model = None
        self.score = 0.0
        self.thres = 0.5 #for binary classification
    
    def get_best_thres(self):
        tmp_pred = self.model.predict(self.tr_data)
        tmp = 0
        for t in range(0,100):
            tmp_thres = t / 100.0
            pred = [int(i > tmp_thres) for i in tmp_pred]
            fs = f1_score(self.tr_label, pred)
            if tmp < fs:
                tmp = fs
                self.thres = tmp_thres
    
    def set_tr_data(self, data):
        self.tr_data = data
    
    def train(self):
        if self.tr_data == None:
            print ("lack of train data")
            return False
        lgb_train = lgb.Dataset(self.tr_data, self.tr_label)
        cv_results = lgb.cv(
                            params = self.params,
                            train_set = lgb_train,
                            seed = self.seed,
                            nfold = self.nfold,
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = self.early_stopping_rounds,
                            verbose_eval = self.verbose_eval
                           )
        if self.flag == -1:
            boost_rounds = pd.Series(cv_results[self.params['metric'] + '-mean']).idxmax()
            self.score = pd.Series(cv_results[self.params['metric'] + '-mean']).max()
        else:
            boost_rounds = pd.Series(cv_results[self.params['metric'] + '-mean']).idxmin()
            self.score = pd.Series(cv_results[self.params['metric'] + '-mean']).min()
        self.model = lgb.train(self.params, lgb_train, num_boost_round = boost_rounds)
        return True
        
    def set_default_params(self):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric':'binary_logloss',
            'num_leaves': 50,
            'learning_rate': 0.01,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'bagging_seed': 55,
            'seed': 77,
            'max_bin': 255,
            
            'nthread': -1,
            'max_depth': -1,
            'verbose': 0
        }
    def reset_min_merror():
        self.min_merror = float('Inf')

    def find_best_params(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        if self.tr_data == None:
            print ("lack of train data")
            return False
        self.adj_leaves_depth(seed, nfold, early_stopping_rounds)
        self.adj_bin_leafdata(seed, nfold, early_stopping_rounds)
        self.adj_fraction(seed, nfold, early_stopping_rounds)
        self.adj_lambda(seed, nfold, early_stopping_rounds)
        return True
    
    def adj_leaves_depth(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for num_leaves in range(20,200,10):
            for max_depth in range(3,8,1):
                params['num_leaves'] = num_leaves
                params['max_depth'] = max_depth 
                cv_results = lgb.cv(
                                    self.params,
                                    self.tr_data,
                                    seed = seed,
                                    nfold = nfold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                if self.flag == -1:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).min()

                if mean_merror * self.flag < self.min_merror * self.flag:
                    self.min_merror = mean_merror
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
        self.params.update(best_params)
        
    def adj_bin_leafdata(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for max_bin in range(100,255,10):
            for min_data_in_leaf in range(10,200,10):
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                cv_results = lgb.cv(
                                    self.params,
                                    self.tr_data,
                                    seed = seed,
                                    nfold = nfold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                if self.flag == -1:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).min()

                if mean_merror * self.flag < self.min_merror * self.flag:
                    self.min_merror = mean_merror
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
        self.params.update(best_params)
        
    def adj_fraction(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for feature_fraction in [0.6,0.7,0.8,0.9]:
            for bagging_fraction in [0.6,0.7,0.8,0.9]:
                cv_results = lgb.cv(
                                    self.params,
                                    self.tr_data,
                                    seed = seed,
                                    nfold = nfold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                if self.flag == -1:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).min()

                if mean_merror * self.flag < self.min_merror * self.flag:
                    self.min_merror = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
        self.params.update(best_params)
    
    def adj_lambda(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for lambda_l1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
            for lambda_l2 in [0.0,0.2,0.4,0.6,0.8,1.0]:
                for min_split_gain in [0.0,1.0]:
                    params['lambda_l1'] = lambda_l1
                    params['lambda_l2'] = lambda_l2
                    params['min_split_gain'] = min_split_gain
                    cv_results = lgb.cv(
                                        self.params,
                                        self.tr_data,
                                        seed = seed,
                                        nfold = nfold,
                                        early_stopping_rounds = early_stopping_rounds,
                                        verbose_eval = 0
                                        )
                if self.flag == -1:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.params['metric'] + '-mean']).min()

                if mean_merror * self.flag < self.min_merror * self.flag:
                        self.min_merror = mean_merror
                        best_params['lambda_l1'] = lambda_l1
                        best_params['lambda_l2'] = lambda_l2
                        best_params['min_split_gain'] = min_split_gain
        self.params.update(best_params)
                
    # TODO: LGB粒子群调参法
    def pso_find_best_params(self):
        pass
        
    def save_model(self, model_path):
        joblib.dump(self.model,model_path)
    
    def load_model(self, model_path):
        self.model = joblib.load(model_path) 

""" 多分类参数 (lgb 在很多类别的时候，分类效果不好)
params = {
          'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'metric': 'multi_logloss', 
          'num_class':3,
          }
"""