# coding: utf-8

import pandas as pd
import numpy as np
import math
from scipy import stats
import sklearn.preprocessing as preproc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import jieba
import jieba.posseg as pseg
import sklearn.preprocessing.OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import FeatureHasher

# # 数值特征

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


# 文本特征
# n元词袋
class BagOfNGram(object):
    def __init__(self, texts, config, stopwordpath = ""):
       if len(stopwordpath) > 0:
            config['stop_words'] = _read_stop_word(stopwordpath)
        self.cv = CountVectorizer(**config)
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
    
    def get_default_config(is_eng = True):
        if is_eng:
            config = {
                'analyzer' : 'word', # 'char', 'char_wb', callable
                'preprocessor' : None, # callable
                'tokenizer' : None, # callable
                'ngram_range' : (1,3),
                'stop_words' : 'english',
                'lowercase' : True,
                'token_pattern' : '(?u)\\b\\w+\\b', #过滤规则，正则表达式，需要设置analyzer == 'word'
                'max_df' : 1.0, # df是document frequency，当df＞[0.0, 1.0]某个值，就过滤掉．也可以是int，表示词出现的次数
                'min_df' : 0.0, # 类似max_df
                'max_features' : None, # 默认为None，可设为int，按frequency降序排序，取前max_features个作为关键词表
                'vocabulary' : None # 指定关键词集
            }
        else:
            config = {
                'analyzer' : 'word', # 'char', 'char_wb', callable
                'preprocessor' : None, # callable
                'tokenizer' : None, # callable
                'ngram_range' : (1,3),
                'stop_words' : None,
                'lowercase' : True,
                'token_pattern' : '(?u)\\b\\w+\\b', #过滤规则，正则表达式，需要设置analyzer == 'word'
                'max_df' : 1.0, # df是document frequency，当df＞[0.0, 1.0]某个值，就过滤掉．也可以是int，表示词出现的次数
                'min_df' : 0.0, # 类似max_df
                'max_features' : None, # 默认为None，可设为int，按frequency降序排序，取前max_features个作为关键词表
                'vocabulary' : None # 指定关键词集
            }
        return config

    def _read_stop_word(path):
        with open(stopwordpath, 'rb') as fp:
            stopword = fp.read().decode('utf-8')  # 提用词提取
        #将停用词表转换为list
        return stopword.splitlines()

# tf-idf 
class TfIdf(object):
    def __init__(self, texts, config, stopwordpath = ""):
       if len(stopwordpath) > 0:
            config['stop_words'] = _read_stop_word(stopwordpath)
        self.cv = TfidfVectorizer(**config)
        self.cv.fit(texts)

    #　列表形式呈现文章生成的词典
    def get_feature_names(self):
        return self.cv.get_feature_names()

    # 字典形式呈现，key：词，value:词频
    def get_vocabulary(self):
        return self.cv.vocabulary_

    def get_idf(self):
        return self.cv.idf_
    
    # 是将结果转化为稀疏矩阵矩阵的表示方式
    def get_sparse_matrix(self, texts):
        return self.cv.transform(texts).toarray()
    
    def get_default_config(is_eng = True):
        if is_eng:
            config = {
                'analyzer' : 'word', # 'char', 'char_wb', callable
                'preprocessor' : None, # callable
                'tokenizer' : None, # callable
                'ngram_range' : (1,3),
                'stop_words' : 'english',
                'lowercase' : True,
                'token_pattern' : '(?u)\\b\\w+\\b', #过滤规则，正则表达式，需要设置analyzer == 'word'
                'max_df' : 1.0, # df是document frequency，当df＞[0.0, 1.0]某个值，就过滤掉．也可以是int，表示词出现的次数
                'min_df' : 1, # 类似max_df
                'max_features' : None, # 默认为None，可设为int，按frequency降序排序，取前max_features个作为关键词表
                'vocabulary' : None, # 指定关键词集
                'norm' : 'l2', # 'l1', 'l2', None
                'use_idf' : True,
                'smooth_idf' : True,
                'sublinear_tf' : False # replace tf with 1 + log(tf)
            }
        else:
            config = {
                'analyzer' : 'word', # 'char', 'char_wb', callable
                'preprocessor' : None, # callable
                'tokenizer' : None, # callable
                'ngram_range' : (1,3),
                'stop_words' : None,
                'lowercase' : True,
                'token_pattern' : '(?u)\\b\\w+\\b', #过滤规则，正则表达式，需要设置analyzer == 'word'
                'max_df' : 1.0, # df是document frequency，当df＞[0.0, 1.0]某个值，就过滤掉．也可以是int，表示词出现的次数
                'min_df' : 1, # 类似max_df
                'max_features' : None, # 默认为None，可设为int，按frequency降序排序，取前max_features个作为关键词表
                'vocabulary' : None, # 指定关键词集
                'norm' : 'l2', # 'l1', 'l2', None
                'use_idf' : True,
                'smooth_idf' : True,
                'sublinear_tf' : False # replace tf with 1 + log(tf)
            }
        return config

    def _read_stop_word(path):
        with open(stopwordpath, 'rb') as fp:
            stopword = fp.read().decode('utf-8')  # 提用词提取
        #将停用词表转换为list
        return stopword.splitlines()




# 切词
# jieba tools
class JiebaTools(object):
    def __init__(self):
        pass

    def get_words(s, cut_all = False, sep = ' '):
        # cut all为True为全模式，各种词都会匹配出来，不能解决歧义
        # cut all为False为精确模式，适合文本分析
        return sep.join(jieba.cut(s, cut_all = cut_all))

    def get_words_search(s, sep = ' '):
        # 搜索引擎模式
        return sep.join(jieba.cut_for_search(s))

    def load_userdict(file_name):
        # 输入格式: 一行一个词，词语、词频(可省略)、词性(可省略)
        jieba.load_userdict(file_name)

    # 动态修改词典
    def add_word(word, freq = None, tag = None):
        jieba.add_word(word, freq, tag)

    def del_word(word):
        jieba.del_word(word)

    # 调节词频
    # 拆开两个字 AB -> A B
    def break_word(A, B):
        jieba.suggest_freq((A, B), True)
    # 避免被拆开 A B -> AB
    def combine(word):
        jieba.suggest_freq(word, True)

    def get_word_pseg(sent):
        words = pseg.cut(sent)
        result = []
        for word, flag in words:
            result.append((word, flag))
        return result

# nltk tools
class NltkTools(object):
    def __init__(self):
        self.stemmer = nltk.stem.porter.PorterStemmer()

    # 词干提取
    def get_stem(self, word):
        return self.stemmer.stem(word)

class DiscreteEncoder(object):
    def __init__(self):
        self.le = LabelEncoder()
        self.oh = OneHotEncoder()

    # label encoder
    def fit_label_encoder(self, data):
        self.le.fit(data)
        return self.label_coding(data)

    def label_coding(self, data):
        return self.le.transform(data)

    def label_decoding(self, data):
        return self.le.inverse_transform(data)

    def get_label_classes(self):
        return self.le.classes_

    def fit_onehot_encoder(self, data):
        self.oh.fit(data)
        return self.onehot_coding(data)

    def onehot_coding(self, data):
        return self.oh.transform(data).toarray()

    def virtual_coding(self):
        # TODO
        pass

    def effect_coding(self):
        # TODO
        pass

# 处理大型分类变量

# 特征散列化 通常用于线性模型
def hash_features(word_list, m):
    output = [0] * m
    for word in word_list:
        index = hash(word) % m
        output[index] += 1
    return output

# 带正负号的特征散列化
# 确保散列后的特征之间的内积等于初始特征内积的期望
def hash_features(word_list, m):
    output = [0] * m
    for word in word_list:
        index = hash(word) % m
        sign_bit = hash(word) % 2
        if (sign_bit == 0):
            output[index] -= 1
        else:
            output[index] += 1
    return output

class FeatureHash(object):
    # string -> m维向量
    def __init__(self, m):
        self.fh = FeatureHasher(n_features = m, input_type = 'string')
        self.hv = HashingVectorizer(n_features = m)

    def fh_transform(self, data):
        return self.fh.transform(data).toarray()
    
    def hv_transform(self, data):
        return self.hv.transform(data).toarray()

# 分箱计数
# 用户在某广告的的优势比 用户 ctr / 此广告其他用户的ctr(可以取log)
