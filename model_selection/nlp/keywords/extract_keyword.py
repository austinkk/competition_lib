#!/usr/bin/env python
# coding: utf-8
import math
import jieba
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
import numpy as np

# textrank不需要训练

# 关键词提取
# 训练
# 1 加载已有的文档数据集
# 2 加载停用词表
# 3 对数据集中的文档进行分词
# 4 根据停用词表，过滤干扰词
# 5 根据数据集训练算法
# 预测
# 对新文档进行分词
# 根据停用词表，过滤干扰词
# 根据训练好的算法提取关键词

# 加载停用词表
def get_stopword_list(stop_word_path):
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path).readlines()]
    return stopword_list

# 切词
def seg_to_list(sentence, pos = False):
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list

# 去除干扰词
def word_filter(seg_list, stop_word_path, pos = False):
    stopword_list = get_stopword_list(stop_word_path)
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)
    return filter_list

def load_data(stop_word_path, pos = False, corpus_path = './corpus.txt'):
    doc_list = []
    for line in open(corpus_path, 'r'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, stop_word_path, pos)
        doc_list.append(filter_list)
    return doc_list

def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    tt_count = len(doc_list)
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
    
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count/(1.0 + v))
    
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf

def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

class TfIdf(object):
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num
    
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0
        
        tt_count = len(self.word_list)
        for k,v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count
        return tf_dic
    
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)
            
            tfidf = tf * idf
            tfidf_dic[word] = tfidf
        
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print (k + '/', end = '')

class TopicModel(object):
    # params: doc_list, keyword_num, model(LSI,LDA)
    def __init__(self, doc_list, keyword_num, model = 'LSI', num_topics = 4):
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择要加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        
        # 得到数据集的主题-词分布
        word_list = [word for doc in doc_list for word in set(doc)]
        self.wordtopic_dic = self.get_wordtopic(word_list)
    
    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics = self.num_topics)
        return lsi
    
    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics = self.num_topics)
        return lda

    def get_wordtopic(self, word_list):
        wordtopic_dic = {}
        for word in word_list:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]
        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x2
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim
        
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim
        
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse = True)[:self.keyword_num]:
            print (k + "/", end='')