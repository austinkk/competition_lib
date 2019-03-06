# -*- coding: utf-8 -*-
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
# from matplotlib import pylab as plt
from sklearn.manifold import TSNE
import pickle
import gc

class SKIPGRAMConfig(object):
    """Skipgram params"""
    filename = './data/en_corpus'
    vocabulary_size = 50000
    batch_size = 128
    embedding_size = 128 # Dimension of the embedding vector.
    skip_window = 1 # How many words to consider left and right.
    num_skips = 2 # How many times to reuse an input to generate a label.
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.    
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100 # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))

    num_sampled = 64 # Number of negative examples to sample.
    num_steps = 100001

    w2v_filename = './tf_%s_num_skip-%s_windows-%s_vocabulary-%s.w2v' % ('SKIPGRAM', num_skips, skip_window * 2 + 1, vocabulary_size)

class SKIPGRAM(object):
    """SKIPGRAM"""
    def __init__(self, config):
        self.data_index = 0
        self.config = config

    def read_data(self, filename, is_zipfile = False):
        """Extract the first file enclosed in a zip file as a list of words"""
        if is_zipfile:
            with zipfile.ZipFile(self.config.filename) as f:
                data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        else:
            with open(self.config.filename) as f:
                data = tf.compat.as_str(f.read()).split()
        return data

    def build_dataset(self, words):
        count = [['UNK', -1]]
        tongji = collections.Counter(words)
        print ('Word types %d' % len(tongji))
        print ('Vocabulary size %d' % self.config.vocabulary_size)
        count.extend(tongji.most_common(self.config.vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
        return data, count, dictionary, reverse_dictionary

    def generate_batch(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0  # each word pair is a batch, so a training data [context target context] would increase batch number of 2.
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[self.config.data_index])
            self.config.data_index = (self.config.data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[self.config.data_index])
            self.config.data_index = (self.config.data_index + 1) % len(data)
        return batch, labels

    def skipgram(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.config.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.config.batch_size, 1])
            valid_dataset = tf.constant(self.config.valid_examples, dtype=tf.int32)

            # Variables.
            embeddings = tf.Variable(
                                     tf.random_uniform([self.config.vocabulary_size, self.config.embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(
                                     tf.truncated_normal([self.config.vocabulary_size, self.config.embedding_size], stddev=1.0 / math.sqrt(self.config.embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([self.config.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)
            # Compute the softmax loss, using a sample of the negative labels each time.
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, self.train_labels, embed, self.config.num_sampled, self.config.vocabulary_size))

            # Optimizer.
            # Note: The optimizer will optimize the softmax_weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities 
            # that contribute to the tensor it is passed.
            # See docs on `tf.train.Optimizer.minimize()` for more details.
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                                            self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, tf.transpose(self.normalized_embeddings))
            embeddings_2 = (self.normalized_embeddings + softmax_weights)/2.0
            norm_ = tf.sqrt(tf.reduce_sum(tf.square(embeddings_2), 1, keep_dims=True))
            self.normalized_embeddings_2 = embeddings_2 / norm_

    def train(self):
        words = self.read_data(self.config.filename)
        print ('Data size %d' % len(words))
        data, count, dictionary, reverse_dictionary = self.build_dataset(words)
        print ('Most common words (+UNK)', count[:5])
        print ('Sample data', data[:10])
        del words  # Hint to reduce memory.
        gc.collect()

        for num_skips, skip_window in [(2, 1), (4, 2)]:
            self.config.data_index = 0
            batch, labels = self.generate_batch(data, batch_size=8, num_skips=num_skips, skip_window=skip_window)
            print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
            print('    batch:', [reverse_dictionary[bi] for bi in batch])
            print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

        self.skipgram()
        with tf.Session(graph=self.graph) as session:
            if int(tf.VERSION.split('.')[1]) > 11:
                tf.global_variables_initializer().run()
            else:
                tf.initialize_all_variables().run()
            print ('Initialized')

            average_loss = 0.0
            for step in range(self.config.num_steps):
                train_batch, train_labels = self.generate_batch(data, self.config.batch_size, self.config.num_skips, self.config.skip_window)
                feed_dict = {self.train_dataset: train_batch, self.train_labels: train_labels}
                l, _ = session.run([self.loss, self.optimizer], feed_dict=feed_dict)
                average_loss += l

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000.0
                    print ('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
                if step % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in range(self.config.valid_size):
                        valid_word = reverse_dictionary[self.config.valid_examples[i]]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]  # let alone itself, so begin with 1
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)

            final_embeddings = self.normalized_embeddings.eval()
            final_embeddings_2 = self.normalized_embeddings_2.eval()  # this is better

        with open(self.config.w2v_filename, 'w') as f:
            f.write('%s %s\n' % (self.config.vocabulary_size, self.config.embedding_size))
            for i in range(self.config.vocabulary_size):
                l = [str(item) for item in list(final_embeddings_2[i])]
                f.write('%s %s\n' % (reverse_dictionary[i], ' '.join(l)))

        """
        num_points = 400

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
        two_d_embeddings_2 = tsne.fit_transform(final_embeddings_2[1:num_points+1, :])

        with open('2d_embedding_cbow.pkl', 'wb') as f:
            pickle.dump([two_d_embeddings, two_d_embeddings_2, reverse_dictionary], f)

        """

if __name__ == '__main__':
    config = SKIPGRAMConfig()
    skipgram_model = SKIPGRAM(config)
    skipgram_model.train()

