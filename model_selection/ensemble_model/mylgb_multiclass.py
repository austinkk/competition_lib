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

        if "df_train" in config:
            self.df_train = config["df_train"]
        else:
            self.df_train = None
        if "df_test" in config:
            self.df_test = config["df_test"]
        else:
            self.tr_label = None

        self.id_name = config["id_name"]
        self.label_name = config["label_name"]
        if "base_feature" in config:
            self.base_feature = config["base_feature"]
        else:
            self.base_feature = [col_name for col_name in self.df_train.columns if col_name not in [self.id_name, self.label_name]]

        if "best_feature" in config:
            self.best_feature = config["best_feature"]
        else:
            self.best_feature = [col_name for col_name in self.base_feature]

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

    def find_best_feature_forward(self):
        base_best_score = self.get_score(self.base_feature)
        while (True):
            is_found = False
            tmp_best_feature = [col_name for col_name in self.base_feature]
            for tmp_feature in itertools.combinations(self.base_feature, len(self.base_feature) - 1):
                tmp_score = self.get_score(list(tmp_feature))
                if tmp_score <= base_best_score:
                    is_found = True
                    tmp_best_feature = [col_name for col_name in list(tmp_feature)]
                    base_best_score = tmp_score
            if is_found:
                self.base_feature = tmp_best_feature
            else:
                break
        self.best_feature = [col_name for col_name in self.base_feature]

    def find_best_feature_backward(self, new_feature_name_list):
        base_best_score = self.get_score(self.base_feature)
        nf_l = new_feature_name_list
        while (True):
            is_found = False
            add_feature = None
            for tmp_feature in nf_l:
                tmp_score = self.get_score(self.base_feature + [tmp_feature])
                if tmp_score <= base_best_score:
                    is_found = True
                    add_feature = tmp_feature
                    base_best_score = tmp_score
            if is_found:
                self.base_feature.append(add_feature)
                nf_l.remove(add_feature)
            else:
                break
        self.best_feature = [col_name for col_name in self.base_feature]

    def get_score(self, fn):
        lgb_train = lgb.Dataset(self.df_train[fn].values, self.df_train[self.label_name].values)
        cv_results = lgb.cv(
                            params = self.params,
                            train_set = lgb_train,
                            seed = self.seed,
                            nfold = self.nfold,
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = self.early_stopping_rounds,
                            verbose_eval = 0
                           )
        boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()
        score = pd.Series(cv_results['multi_logloss-mean']).min()
        return score

    def train(self, make_submit = False, filepath = None):
        lgb_train = lgb.Dataset(self.df_train[self.best_feature].values, self.df_train[self.label_name].values)
        cv_results = lgb.cv(
                            params = self.params,
                            train_set = lgb_train,
                            seed = self.seed,
                            nfold = self.nfold,
                            num_boost_round = self.num_boost_round,
                            early_stopping_rounds = self.early_stopping_rounds,
                            verbose_eval = self.verbose_eval
                           )
        boost_rounds = pd.Series(cv_results['multi_logloss-mean']).idxmin()
        score = pd.Series(cv_results['multi_logloss-mean']).min()
        self.model = lgb.train(self.params, lgb_train, num_boost_round = boost_rounds)
        if make_submit:
            self.make_submit(filepath)
        return score

    def make_submit(self, filepath):
        test_y = self.model.predict(self.df_test[self.best_feature].values)
        test_y = np.array(test_y, dtype = int)
        df_submit = pd.DataFrame()
        df_submit['id'] = self.df_test[self.id_name].values
        df_submit['score'] = test_y
        if filepath == None:
            tmp = './submit/submit_%s.csv' % int(time.time())
            df_submit.to_csv(tmp, index = None)
        else:
            df_submit.to_csv(filepath, index = None)

        
    def set_default_params(self):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric':'multi_logloss',
            'num_class':4,
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

    def find_best_params(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        self.adj_depth(seed, nfold, early_stopping_rounds)
        print ('has find best max_depth:%s' % self.params['max_depth'])
        self.adj_min_child_weight(seed, nfold, early_stopping_rounds)
        print ('has find best min_child_weight:%s' % self.params['min_child_weight'])
        #self.adj_bin_leafdata(seed, nfold, early_stopping_rounds)
        self.adj_fraction(seed, nfold, early_stopping_rounds)
        print ('has find best feature_fraction/bagging_fraction:%s/%s' % (self.params['feature_fraction'],self.params['bagging_fraction'] ))
        self.adj_lambda(seed, nfold, early_stopping_rounds)
        print ('has find best lambda_l1/lambda_l2/min_split_gain:%s/%s/%s' % (self.params['lambda_l1'], self.params['lambda_l2'],self.params['min_split_gain']))
        #self.adj_eta(seed, nfold, early_stopping_rounds)
        return True

    def adj_eta(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        lgb_train = lgb.Dataset(self.tr_data, self.tr_label)
        for eta in [0.01, 0.015, 0.025, 0.05, 0.1]:
            self.params['learning_rate'] = eta
            cv_results = lgb.cv(
                                self.params,
                                lgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['learning_rate'] = eta
        self.params.update(best_params)

    def adj_depth(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        lgb_train = lgb.Dataset(self.df_train[self.best_feature].values, self.df_train[self.label_name].values)
        for max_depth in [3,5,7,9,12,15,17,25]:
            self.params['max_depth'] = max_depth
            cv_results = lgb.cv(
                                self.params,
                                lgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['max_depth'] = max_depth
        self.params.update(best_params)

    def adj_min_child_weight(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        lgb_train = lgb.Dataset(self.df_train[self.best_feature].values, self.df_train[self.label_name].values)
        for min_child_weight in [1, 3, 5, 7]:
            self.params['min_child_weight'] = min_child_weight
            cv_results = lgb.cv(
                                self.params,
                                lgb_train,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()

            if mean_merror  <= self.min_merror:
                self.min_merror = mean_merror
                best_params['min_child_weight'] = min_child_weight
        self.params.update(best_params)

    def adj_bin_leafdata(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        lgb_train = lgb.Dataset(self.df_train[self.best_feature].values, self.df_train[self.label_name].values)
        for max_bin in range(100,255,10):
            for min_data_in_leaf in range(10,200,10):
                self.params['max_bin'] = max_bin
                self.params['min_data_in_leaf'] = min_data_in_leaf
                cv_results = lgb.cv(
                                    self.params,
                                    lgb_train,
                                    seed = seed,
                                    nfold = nfold,
                                    num_boost_round = self.num_boost_round,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()

                if mean_merror  <= self.min_merror:
                    self.min_merror = mean_merror
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
        self.params.update(best_params)

    def adj_fraction(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        lgb_train = lgb.Dataset(self.df_train[self.best_feature].values, self.df_train[self.label_name].values)
        for feature_fraction in [0.6,0.7,0.8,0.9, 1]:
            for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1]:
                self.params['feature_fraction'] = feature_fraction
                self.params['bagging_fraction'] = bagging_fraction
                cv_results = lgb.cv(
                                    self.params,
                                    lgb_train,
                                    seed = seed,
                                    nfold = nfold,
                                    num_boost_round = self.num_boost_round,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()

                if mean_merror <= self.min_merror:
                    self.min_merror = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
        self.params.update(best_params)

    def adj_lambda(self, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        lgb_train = lgb.Dataset(self.df_train[self.best_feature].values, self.df_train[self.label_name].values)
        for lambda_l1 in [0.0, 0.1, 0.5, 1.0]:
            for lambda_l2 in [0.0, 0.01, 0.1, 1.0]:
                for min_split_gain in [0.0,1.0]:
                    self.params['lambda_l1'] = lambda_l1
                    self.params['lambda_l2'] = lambda_l2
                    self.params['min_split_gain'] = min_split_gain
                    cv_results = lgb.cv(
                                        self.params,
                                        lgb_train,
                                        seed = seed,
                                        nfold = nfold,
                                        num_boost_round = self.num_boost_round,
                                        early_stopping_rounds = early_stopping_rounds,
                                        verbose_eval = 0
                                        )
                mean_merror = pd.Series(cv_results['multi_logloss-mean']).min()

                if mean_merror <= self.min_merror:
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
