import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
from minepy import MINE
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE

def removing_features_with_low_variance(df, var_thres = 0, black_list = []):
    suggest_remove_list = []
    for i in df.columns:
        if i in black_list:
            continue
        if i.startswith('ls_'):
            md = df[i].mode()
            md_num = sum(raw_train_data[i].values == md[0])
            if md_num * 1 / len(df) >= 0.95:
                suggest_remove_list.append(i)
        else:
            if df[i].var() < var_thres:
                suggest_remove_list.append(i)
    return suggest_remove_list

def pearson_correlation(df, label, thres = 0, black_list = [], allow_ls = True):
    #-1表示完全的负相关(这个变量下降，那个就会上升)，+1表示完全的正相关，0表示没有线性相关。
    #只对线性关系敏感。如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0
    suggest_remove_list = []
    d = {}
    for i in df.columns:
        if i in black_list:
            continue
        if i.startswith('ls_') and not allow_ls:
            continue
        d[i] = pearsonr(df[i].values, label)[0]
        if abs(d[i]) <= thres:
            suggest_remove_list.append(i)
    return d, suggest_remove_list

def model_based_ranking(df, label, is_classification = True, black_list = []):
    d = {}
    if is_classification:
        rf = RandomForestClassifier(n_estimators=20, max_depth=4)
    else:
        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    for i in df.columns:
        if i in black_list:
            continue
        X =[[item] for item in df[i].values]
        score = cross_val_score(rf, X, label, scoring="r2", cv=ShuffleSplit(len(X), 3, .3))
        d[i] = round(np.mean(score), 3)
    return d

def linear_regression_weight(df, label, black_list = []):
    X = df.drop(black_list, axis = 1)
    lr = LinearRegression()
    lr.fit(X.values, label)
    d = dict(zip(X.columns, lr.coef_))
    return d

def lasso_l1_weight(df, label, black_list = []):
    X = df.drop(black_list, axis = 1)
    scaler = StandardScaler()
    X_v = scaler.fit_transform(X.values)
    lasso = Lasso(alpha=.3)
    lasso.fit(X_v, label)
    d = dict(zip(X.columns, lasso.coef_))
    return d

def ridge_l2_weight(df, label, black_list = []):
    X = df.drop(black_list, axis = 1)
    ridge = Ridge(alpha=10)
    ridge.fit(X.values,label)
    d = dict(zip(X.columns, ridge.coef_))
    return d

def mean_decrease_impurity(df, label, is_classification = True, black_list = []):
    if is_classification:
        rf = RandomForestClassifier()
    else:
        rf = RandomForestRegressor()
    X = df.drop(black_list, axis = 1)
    rf = RandomForestRegressor()
    rf.fit(X.values, label)
    d = dict(zip(X.columns, rf.feature_importances_))
    return d

def mean_decrease_accuracy_regression(df, Y, black_list = []):
    #直接度量每个特征对模型精确率的影响。主要思路是打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。
    #很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率
    rf = RandomForestRegressor()
    scores = defaultdict(list)
    X_src = df.drop(black_list, axis = 1)
    X = X_src.values
    names = X_src.columns
    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc-shuff_acc)/acc)
    return dict([(round(np.mean(score), 4), feat) for feat, score in scores.items()])

def linear_regression_weight(df, label, black_list = []):
    #稳定性选择是一种基于二次抽样和选择算法相结合较新的方法，选择算法可以是回归、SVM或其他类似的方法。
    #它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
    #比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。
    #理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。
    X = df.drop(black_list, axis = 1)
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(X.values, label)
    d = dict(zip(X.columns, rlasso.scores_))
    return d

def recursive_feature_elimination(df, label, black_list = []):
    #递归特征消除的主要思想是反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），
    #把选出来的特征放到一遍，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。
    #这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。
    X = df.drop(black_list, axis = 1)
    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X.values, label)
    d = dict(zip(X.columns, rfe.ranking_))
    return d

