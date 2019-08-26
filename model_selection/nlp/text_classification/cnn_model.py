# coding: utf-8

from __future__ import print_function

import tensorflow as tf
import os
import sys
import time
from datetime import timedelta
import tensorflow.contrib.keras as kr
import numpy as np
from sklearn import metrics

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    train_path = './tfrecord/train.record' # 训练集 tfrecord 
    valid_path = './tfrecord/valid.record' # 验证机 tfrecord
    test_path = './tfrecord/valid.record' # 测试集 tfrecord

    tensorboard_dir = 'tensorboard/textcnn'

    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        if not os.path.exists(self.config.train_path) or not os.path.exists(self.config.valid_path):
            raise ValueError("""tfrecord file not exist""")

        if not os.path.exists(config.tensorboard_dir):
            os.makedirs(config.tensorboard_dir)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int64, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int64, None, name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.prob, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            input_y_onehot = tf.one_hot(self.input_y, self.config.num_classes, 1, 0)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=input_y_onehot)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(self.input_y, self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds = int(round(time_dif)))

    def _feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.keep_prob: keep_prob
        }
        return feed_dict

    def _evaluate(self, session, filepath):
        """评估在某一数据上的准确率和损失"""
        def model(seq, labels):
            return seq, labels
        val_dataset = model(*self._batched_data(filepath, self._single_example_parser, self.config.batch_size, ([None], []), 1, 1))
        data_len = 0
        total_loss = 0.0
        total_acc = 0.0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        try:
            while not coord.should_stop():
                x_val_batch, y_val_batch = session.run(val_dataset)
                x_val_batch = kr.preprocessing.sequence.pad_sequences(x_val_batch, self.config.seq_length)
                batch_len = len(x_val_batch)
                data_len += batch_len
                feed_dict = self._feed_data(x_val_batch, y_val_batch, self.config.dropout_keep_prob)
                feed_dict[self.keep_prob] = 1.0
                loss, acc = session.run([self.loss, self.acc], feed_dict=feed_dict)
                total_loss += loss * batch_len
                total_acc += acc * batch_len
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)
        return total_loss / data_len, total_acc / data_len

    def _single_example_parser(self, serialized_example):
        context_features = {
            'label' : tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {}
        for i in range(1):
            sequence_features['seq' + str(i)] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features = context_features,
                sequence_features = sequence_features
        )
        seq0 = sequence_parsed['seq0']
        labels = context_parsed['label']
        return seq0, labels

    def _batched_data(self, tfrecord_filename, single_example_parser, batch_size, padded_shapes, num_epochs = 1, buffer_size = 1000):
        dataset = tf.data.TFRecordDataset(tfrecord_filename) \
                    .map(single_example_parser) \
                    .shuffle(buffer_size) \
                    .padded_batch(batch_size, padded_shapes=padded_shapes) \
                    .repeat(num_epochs)
        return dataset.make_one_shot_iterator().get_next()

    def train(self):
        print("Configuring TensorBoard and Saver...")
        # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.config.tensorboard_dir)

        # 配置 Saver
        saver = tf.train.Saver()

        # 创建session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        
        def model(seq, labels):
            return seq, labels
        tr_dataset = model(*self._batched_data(self.config.train_path, self._single_example_parser, self.config.batch_size, ([None], []), self.config.num_epochs))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        try:
            while not coord.should_stop():
                x_batch, y_batch = session.run(tr_dataset)
                x_batch = kr.preprocessing.sequence.pad_sequences(x_batch, self.config.seq_length)
                feed_dict = self._feed_data(x_batch, y_batch, self.config.dropout_keep_prob)

                if total_batch % self.config.save_per_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                if total_batch % self.config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[self.keep_prob] = 1.0
                    loss_train, acc_train = session.run([self.loss, self.acc], feed_dict=feed_dict)
                    loss_val, acc_val = self._evaluate(session, self.config.valid_path)

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=self.config.save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = self._get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(self.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1

                if total_batch - last_improved > self.config.require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    break
        except tf.errors.OutOfRangeError:
            print("done training")
        finally:
            coord.request_stop()
        coord.join(threads)

    def test(self, filepath, is_dump_metrics = False, is_dump_result = False, dump_filename = './pred/result'):
        print("Loading test data...")
        start_time = time.time()
        
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=self.config.save_path)  # 读取保存的模型

        """
        print("Testing...")
        loss_test, acc_test = evaluate(session, filepath)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))
        """

        def model(seq, labels):
            return seq, labels
        test_dataset = model(*self._batched_data(filepath, self._single_example_parser, self.config.batch_size, ([None], []), 1, 1))
        data_len = 0

        y_test_cls = []
        y_pred_cls = []
        y_pred_prob_cls = []

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        try:
            while not coord.should_stop():
                x_test_batch, y_test_batch = session.run(test_dataset)
                y_test_cls.extend(list(y_test_batch))
                x_test_batch = kr.preprocessing.sequence.pad_sequences(x_test_batch, self.config.seq_length)
                feed_dict = self._feed_data(x_test_batch, y_test_batch, self.config.dropout_keep_prob)
                feed_dict[self.keep_prob] = 1.0

                y_pred, y_prob = session.run([self.y_pred_cls, self.prob], feed_dict=feed_dict)
                y_pred_cls.extend(list(y_pred))
                y_pred_prob_cls.extend(list(y_prob))

        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)

        y_test_cls = np.array(y_test_cls)
        y_pred_cls = np.array(y_pred_cls)
        y_pred_prob_cls = np.array(y_pred_prob_cls)
       
        # 评估
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)
        if is_dump_metrics:
            np.savetxt('%s.metrics' % dump_filename, cm)
        if is_dump_result:
            np.savetxt('%s.prob' % dump_filename, y_pred_prob_cls)
            np.savetxt('%s.pred' % dump_filename, y_pred_cls)
        
        time_dif = self._get_time_dif(start_time)
        print("Time usage:", time_dif)
