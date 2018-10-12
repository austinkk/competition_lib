#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn import linear_model
from sklearn.grid_search import GridSearchCV

class Mylinear(object):
    """mylr is best tool for data mining"""
    def __init__(self, config):
        if  "params" in config:
            self.params = config["params"]
        else:
            self.set_default_params()
        if "tr_data" in config:
            self.tr_data = config["tr_data"]
        else:
            self.tr_data = None
        if "tr_label" in config:
            self.tr_label = config["tr_label"]
        else:
            self.tr_label = None
        self.model = None
        self.score = 0.0

    def set_tr_data(self, data):
        self.tr_data = data
    
    def set_tr_label(self, data):
        self.tr_label = data

    def train(self):
        #solver表示学习参数的办法，有'newton-cg','lbfgs','liblinear','sag','saga'
        #小数据集选liblinear，大数据集选saga,sag
        if self.tr_data == None or self.tr_label = None:
            print ("lack of train data or train label")
            return False
        self.model = linear_model.LogisticRegression(
                                                     penalty = self.params['penalty'],
                                                     C = self.params['C'],
                                                     solver = self.params['solver'],
                                                     n_jobs = -1
                                                    ).fit(self.tr_data, self.tr_label)
        return True
    
    def train_with_cv(self, cv = 5):
        if self.tr_data == None or self.tr_label = None:
            print ("lack of train data or train label")
            return False
        self.model = linear_model.LogisticRegressionCV(
                                                     penalty = self.params['penalty'],
                                                     C = self.params['C'],
                                                     solver = self.params['solver'],
                                                     n_jobs = -1,
                                                     cv = cv
                                                    ).fit(self.tr_data, self.tr_label)
        return True
    
    #TODO: 其他模型没有调参
    def train_with_LassoCV(self):
        if self.tr_data == None or self.tr_label = None:
            print ("lack of train data or train label")
            return False
        self.model = linear_model.LassoCV().fit(self.tr_data, self.tr_label)
        return True
    
    def train_with_RidgeCV(self):
        if self.tr_data == None or self.tr_label = None:
            print ("lack of train data or train label")
            return False
        self.model = linear_model.RidgeClassifierCV().fit(self.tr_data, self.tr_label)
        return True

    def train_with_ElasticNetCV(self):
        if self.tr_data == None or self.tr_label = None:
            print ("lack of train data or train label")
            return False
        self.model = linear_model.MultiTaskElasticNetCV().fit(self.tr_data, self.tr_label)
        return True
        
    def set_default_params(self):
        self.params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver':'lbfgs'
        }
    
    def find_best_params(self, cv = 5):
        C = [0.1, 0.2, 0.5, 0.8, 1.5, 3, 5]
        fit_intercept = [True, False]
        penalty = ['l1', 'l2']
        solver = ['newton-cg','lbfgs','liblinear','sag','saga']
        param_grid = dict(C = C, fit_intercept = fit_intercept, penalty = penalty, solver = solver)
        clf = liblinear.LogisticRegression(n_jobs = -1)
        grid = GridSearchCV(clf, param_grid, cv = cv, scoring = 'accuracy')
        self.params.update(grid.best_estimator_.get_params())
        
        