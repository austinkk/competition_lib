#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
import numpy as np
import collections
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

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

# 标准化/归一化
#归一化算法是通过特征的最大最小值将特征缩放到[0,1]区间范围内，而多于许多机器学习算法，标准化也许会更好，
#标准化是通过特征的平均值和标准差将特征缩放成一个标准的正态分布，均值为0，方差为1。
def get_normalization_maxmin(series):
    minMax = MinMaxScaler()
    x_std = minMax.fit_transform(series)
    return x_std

def get_normalization_maxabs(series):
    maxAbs = MaxAbsScaler()
    x_std = maxAbs.fit_transform(series)
    return x_std

def get_standardization(series):
    ss = StandardScaler()
    x_std = ss.fit_transform(series)
    return x_std

#y = atan(x) * 2 / π
def get_atan_2pi(df, f_name):
    df[f_name] = df[f_name].apply(lambda x: math.atan(x) * 2 / math.pi)
