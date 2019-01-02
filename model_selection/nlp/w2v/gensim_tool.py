# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

class W2VConfig(object):
    """w2v parameters"""
    sg = 0 # sg: 0 is CBOW 1 is Skip-gram
    size = 128 # dimension of word
    window = 5
    min_count = 5 # the word will be filtered when occuring times less than threshold
    workers = 9 # threads num
    addname = True
    cbow_mean = 1 # 1 mean of context 0 sum of context !only work on cbow!
    iter = 5

    src_path = ""
    save_dir = "./"
    save_filename = ""

    save_model = False

class Gensim_tool(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    def generate_w2v(self):
        is_file = os.path.exists(self.config.src_path)
        is_dir = os.path.exists(self.config.save_dir)
        if not is_file:
            print ('source file not exist')
            return False
        if not is_dir:
            print ('save path not exist')
            return False
        corpus = open(self.config.src_path, 'r')
        #for line in LineSentence(corpus):
        #    print (line)
        self.model = Word2Vec(LineSentence(corpus), sg = self.config.sg, size = self.config.size, window = self.config.window, min_count = self.config.min_count, workers = self.config.workers, cbow_mean = self.config.cbow_mean, iter = self.config.iter)
        if self.config.save_model:
            self.model.save(self.get_filename() + '.model')
        #self.model.wv.save(self.get_filename() + '.kv')
        self.model.wv.save_word2vec_format(self.get_filename() + '.w2v')
        return True

    def get_filename(self):
        if not self.config.addname:
            return os.path.join(self.config.save_dir, self.config.save_filename)
        param_info = 'gensim_%s_size-%s_windows-%s_mc-%s_' % ('CBOW' if self.config.sg == 0 else 'SkipGram', self.config.size, self.config.window, self.config.min_count)
        return os.path.join(self.config.save_dir, param_info + self.config.save_filename)

    def load_model(self, path):
        self.model = gensim.models.Word2Vec.load(path)

    def get_similarity(self, w1, w2):
        return self.model.similarity(w1, w2)

    def get_most_similarity(self, word):
        if word in self.model.wv.index2word:
            return self.model.most_similar(word)

if __name__ == '__main__':
    wordconfig = W2VConfig()
    wordconfig.src_path = './data/A_en.txt'
    wordconfig.save_dir = './data'
    wordconfig.save_filename = 'w2v'
    wordconfig.iter = 10
    gensim_tool = Gensim_tool(wordconfig)
    gensim_tool.generate_w2v()
