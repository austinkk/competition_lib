
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from scipy import stats
import sklearn.preprocessing as preproc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# # 数值特征

# In[12]:


# 分箱
def fixed_box(x, box_bound):
    # example: age 0-12 12-17 18-24 25-34 35-44 45-54 55-64 65-74 75+ 
    #　数量级　0-9 10-99 100-999 1000-9999
    result = []
    for i in x:
        idx = 0
        tmp_idx = -1
        for j in box_bound:
            if i < j:
                tmp_idx = idx
                break
            idx += 1
        if tmp_idx == -1:
            tmp_idx = idx
        result.append(tmp_idx)
    return np.array(result)

def average_box(x, box_num):
    return np.array(np.floor_divide(x, box_num))

def log_box(x):
    return np.array(np.floor(np.log10(x)))

def quantile_box(x, q_num):
    interval = 1.0 / q_num
    quantile =[i * interval for i in range(1,q_num)]
    large_counts_series = pd.Series(x)
    print (large_counts_series.quantile(quantile))
    return pd.qcut(x, q_num, labels=False)


# In[28]:


# 对数变换　（指数变换的特例）
def log_trans(x, base):
    # 对数变化后，分布更偏正态分布
    return np.array([math.log(i, base) for i in x])

# 指数变换
# 方差稳定变换,使得方差不再依赖于均值
def box_cox(x, lambda_ = 0, auto = True):
    if auto:
        result, best_lambda =  stats.boxcox(x)
    else:
        result = stats.boxcox(x, lambda_)
    return result


# In[51]:


# 特征缩放，归一化
# 不要＂中心化＂稀疏数据，如果平移量不是０，会导致多数元素为０的稀疏特征向量变成密集特征向量，会给分类器带来巨大负担
# 不会改变分布
#min-max
def min_max(x):
    return preproc.minmax_scale(x)

def standardlized(x):
    df = pd.DataFrame()
    df['tmp'] = x
    return preproc.StandardScaler().fit_transform(df[['tmp']]).reshape(-1)

def l2(x):
    df = pd.DataFrame()
    df['tmp'] = x
    return preproc.normalize(df[['tmp']], axis = 0).reshape(-1)


# # 文本特征

# In[ ]:


# n元词袋


# In[ ]:


class BagOfNGram(object):
    def __init__(self, texts, stopwordpath = "", is_english = True):
        with open(stopwordpath, 'rb') as fp:
            stopword = fp.read().decode('utf-8')  # 提用词提取
        #将停用词表转换为list
        stpwrdlst = stopword.splitlines()
        if is_english:
            self.cv = CountVectorizer(stop_words = stpwrdlst)
        else:
            self.cv = CountVectorizer(stop_words = 'english')
        self.cv.fit(texts)

    #　列表形式呈现文章生成的词典
    def get_feature_names(self):
        return self.cv.get_feature_names()

    # 字典形式呈现，key：词，value:词频
    def get_vocabulary(self):
        return self.cv.vocabulary_
    
    # 是将结果转化为稀疏矩阵矩阵的表示方式
    def get_sparse_matrix(self, texts):
        return self.cv.transform(texts).toarray()
    
    def get_default_config():
        config = {
            analyzer = 'word', # 'char', 'char_wb', callable
            preprocessor = None, # callable
            tokenizer = None, # callable
            ngram_range  = (1,3),
            stop_words = 'english',
            lowercase = True,
            #token_pattern = '' #过滤规则，正则表达式，需要设置analyzer == 'word'
            max_df = 1.0 # df是document frequency，当df＞[0.0, 1.0]某个值，就过滤掉．也可以是int，表示词出现的次数
            min_df = 0.0 # 类似max_df
            max_features = None # 默认为None，可设为int，
        }

