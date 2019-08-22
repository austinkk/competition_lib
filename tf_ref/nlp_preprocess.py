#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import io
import os
import sys
from data_loader import DataLoader

class NlpTFRecorder(object):
    def __init__(self, seq_num, is_reg = True):
        self.seq_num = seq_num
        self.is_reg = is_reg
        self.data_loader = DataLoader()

    def preprocess(self, train_dir, vocab_dir, label_dir, vocab_size):
        self.data_loader.build_vocab(traion_dir, vocab_dir, label_dir, vocab_size)
        _, self.word_to_id, _ = self.data_loader.read_vocab(vocab_dir)
        _, self.label_to_id, _ = self.data_loader.read_label(label_dir)

    def process(self, file_list, file_name_list):
        for file_dir, record_dir in zip(file_list, file_name_list): 
            self.generate_tfrecord(file_dir, record_dir)

    def int64_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
    
    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

    def list_feature(self, value):
        return tf.train.FeatureList(feature = value)

    def create_tf_example(self, sequence, label):
        # feature : [1,2,3,4,2,4,5,1,2] label : num
        feature_list = {}
        idx = 0
        for s in sequence:
            frame_feature = list(map(self.int64_feature, s))
            feature_list['seq' + str(idx)] = self.list_feature(frame_feature)
            idx += 1

        tf_example = tf.train.SequenceExample(
            context = tf.train.Features(
                    feature = {
                        'label' : self.int64_feature(label)
                    }
            ),
            feature_lists = tf.train.FeatureLists(
                feature_list = feature_list
            )
        )
        return tf_example

    def generate_tfrecord(self, data_path, record_path):
        # annotation [([s1, s2, ... , sn], label)]
        num_tf_example = 0
        writer = tf.python_io.TFRecordWriter(record_path)
        for sequence, label in annotation_list:
            tf_example = self.create_tf_example(sequence, label)
            writer.write(tf_example.SerializeToString())
            num_tf_example += 1
            if num_tf_example % 100 == 0:
                print("Create %d TF_Example" % num_tf_example)
        writer.close()
        print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))

    def single_example_parser(self, serialized_example):
        context_features = {
            'label' : tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {}
        for i in range(self.seq_num):
            sequence_features['seq' + str(i)] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features = context_features,
            sequence_features = sequence_features
        )

        labels = context_parsed['label']
        if self.seq_num == 1:
            seq1 = sequence_parsed['seq1']
            return seq1, labels
        if self.seq_num == 2:
            seq2 = sequence_parsed['seq2']
            return seq1, seq2, labels
        if self.seq_num == 3:
            seq3 = sequence_parsed['seq3']
            return seq1, seq2, seq3, labels
        return None

    def batched_data(self, tfrecord_filename, single_example_parser, batch_size, padded_shapes, num_epochs = 1, buffer_size = 1000):
        dataset = tf.data.TFRecordDataset(tfrecord_filename) \
                .map(single_example_parser) \
                .shuffle(buffer_size) \
                .padded_batch(batch_size, padded_shapes=padded_shapes) \
                .repeat(num_epochs)
        return dataset.make_one_shot_iterator().get_next()

    def test(self, tfrecord_filename, pad1 = None, pad2 = None):
        def model(s1, s2, labels):
            return s1, s2, labels
        out = model(*self.batched_data(tfrecord_filename, self.single_example_parser, 2, ([pad1], [pad2] ,[])))
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
        # config.gpu_options.allow_growth = True #动态分配显存
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    print(sess.run(out))
            except tf.errors.OutOfRangeError:
                print("done training")
            finally:
                coord.request_stop()
            coord.join(threads)
