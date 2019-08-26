# coding: utf-8

import sys
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np
import tensorflow.contrib.keras as kr

class DataLoader(object):
    def __init__(self):
        if sys.version_info[0] > 2:
            self.is_py3 = True
        else:
            reload(sys)
            sys.setdefaultencoding("utf-8")
            self.is_py3 = False
    
    
    def native_word(self, word, encoding='utf-8'):
        """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
        if not self.is_py3:
            return word.decode(encoding)
        else:
            return word
    
    
    def native_content(self, content, encoding='utf-8'):
        if not self.is_py3:
            return content.decode('utf-8')
        else:
            return content
    
    
    def open_file(self, filename, mode='r'):
        """
        常用文件操作，可在python2和python3间切换.
        mode: 'r' or 'w' for read or write
        """
        if self.is_py3:
            return open(filename, mode, encoding='utf-8', errors='ignore')
        else:
            return open(filename, mode)
    
    
    def read_file(self, filename):
        """读取文件数据"""
        contents = []
        labels = []
        with self.open_file(filename) as f:
            for line in f:
                try:
                    # content1 \t content2 \t ... \t label
                    ls = line.strip().split('\t')
                    tmp = []
                    for i in range(len(ls) - 1):
                        tmp.append((self.native_content(ls[i])).split(' '))
                    contents.append(tmp)
                    labels.append(self.native_word(ls[-1]))
                except:
                    pass
        return contents, labels
    
    
    def build_vocab(self, train_dir, vocab_dir, label_dir, vocab_size=5000):
        """根据训练集构建词汇表，存储"""
        data_train, data_train_label = self.read_file(train_dir)
    
        all_data = []
        for ins in data_train:
            for content in ins:
                all_data.extend(content)
        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)
        self.open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

        labels = list(set(data_train_label))
        self.open_file(label_dir, mode='w').write('\n'.join(labels) + '\n')


    def read_vocab(self, vocab_dir):
        """读取词汇表"""
        # words = open_file(vocab_dir).read().strip().split('\n')
        with self.open_file(vocab_dir) as fp:
            # 如果是py2 则每个值都转化为unicode
            words = [self.native_content(_.strip()) for _ in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        id_to_word = dict(zip(range(len(words)), words))
        return words, word_to_id, id_to_word
    
    
    def read_label(self, label_dir):
        """读取分类目录，固定"""
        with self.open_file(label_dir) as fp:
            labels = [self.native_content(_.strip()) for _ in fp.readlines()]
    
        label_to_id = dict(zip(labels, range(len(labels))))
        id_to_label = dict(zip(range(len(labels)), labels))   
        return labels, label_to_id, id_to_label
    
    
    def to_words(self, content, id_to_word):
        """将id表示的内容转换为文字"""
        return ' '.join(words[x] for x in content)
