#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
import numpy as np
import collections

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def is_int(x):
    try:
        int(x)
    except:
        return False
    return True

def one_hot(d, idx):
    tmp = [0] * len(d)
    tmp[d[idx]] = 1
    return tmp
    
def calc_fscore(p, r):
    return (2 * p * r) / (p + r)
    
def calc_multi_fscore(pred, label):
    dict_num_class_pred = {}
    dict_num_class_label = {}
    dict_num_cor_class = {}
    for i,j in zip(pred, label):
        dict_num_class_pred.setdefault(i, 0)
        dict_num_class_pred[i] += 1
        dict_num_class_label.setdefault(j, 0)
        dict_num_class_label[j] += 1
        if i == j:
            dict_num_cor_class.setdefault(i, 0)
            dict_num_cor_class[i] += 1
    f1_score_list = []
    for i in dict_num_cor_class:
        p = dict_num_cor_class[i] / dict_num_class_pred[i]
        r = dict_num_cor_class[i] / dict_num_class_label[i]
        f1_score_list.append(calc_fscore(p, r))
    return np.average(f1_score_list)
	
def save_params(params, params_path):
    f = open(params_path,'w')
    f.write(str(params))
    f.close()

def read_params(params_path):
    f = open(params_path,'r')
    tmp = f.read()
    params = eval(tmp)
    f.close()
    return params

def save_model(model, model_path):
    joblib.dump(model,model_path)
    
def load_model(model_path):
    return joblib.load(model_path)

def get_split_data(tr_feature, tr_label, scale = 0.2, random_state = 42):
    X_train,  X_val, y_train, y_val = train_test_split(tr_feature, tr_label, test_size = scale, random_state = random_state)
    return X_train, X_val, y_train, y_val
