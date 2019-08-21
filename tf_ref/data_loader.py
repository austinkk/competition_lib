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
        if not is_py3:
            return word.decode(encoding)
        else:
            return word
    
    
    def native_content(self, content, encoding='utf-8'):
        if not is_py3:
            return content.decode('utf-8')
        else:
            return content
    
    
    def open_file(self, filename, mode='r'):
        """
        常用文件操作，可在python2和python3间切换.
        mode: 'r' or 'w' for read or write
        """
        if is_py3:
            return open(filename, mode, encoding='utf-8', errors='ignore')
        else:
            return open(filename, mode)
    
    
    def read_file(self, filename):
        """读取文件数据 func 用来改变格式"""
        contents1, contents2, labels = [], [], []
        with self.open_file(filename) as f:
            for line in f:
                try:
                    if func != None:
                        line = func(line)
                    # label \t content1 \t content2
                    ls = line.strip().split('\t')
                    if len(ls) == 2:
                        contents1.append((self.native_content(ls[1])).split(' '))
                        labels.append(self.native_content(label))
                    else:
                        contents1.append((self.native_content(ls[1])).split(' '))
                        contents2.append((self.native_content(ls[2])).split(' '))
                        labels.append(self.native_word(label))
                except:
                    pass
        return contents1, contents2, labels
    
    
    def build_vocab(self, train_dir, vocab_dir, label_dir, vocab_size=5000):
        """根据训练集构建词汇表，存储"""
        data_train1, data_train2, data_train_label = self.read_file(train_dir)
    
        all_data = []
        for content in data_train1:
            all_data.extend(content)
        for content in data_train2:
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
        return words, word_to_id
    
    
    def read_category(self):
        """读取分类目录，固定"""
        categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    
        categories = [native_content(x) for x in categories]
    
        cat_to_id = dict(zip(categories, range(len(categories))))
    
        return categories, cat_to_id
    
    
    def to_words(self, content, words):
        """将id表示的内容转换为文字"""
        return ' '.join(words[x] for x in content)
    
    
    def process_file(self, filename, word_to_id, cat_to_id, max_length=600):
        """将文件转换为id表示"""
        contents, labels = read_file(filename)
    
        data_id, label_id = [], []
        for i in range(len(contents)):
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])
    
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    
        return x_pad, y_pad
