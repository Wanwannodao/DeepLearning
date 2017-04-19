#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import tarfile
import urllib.request
import numpy as np
import collections
import random

import tensorflow as tf

# ====================
# ops
# ====================

def LSTM(size, is_tuple=True, prob=1.0):
    cell = None
    if 'reuse' in inspect.getargspec(
            tf.contrib.rnn.BasicLSTMCell.__init__).args:
        cell = tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=is_tuple,
            reuse=tf.get_variable_scope().reuse)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=is_tuple)

    if prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=prob)
    return cell

def fc(x, in_dim, out_dim, bn=False, activation_fn=None, scope="fc"):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", [in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, W), b)
        if bn:
            fc = batch_norm(fc, axes=[0])
            
        if activation_fn is not None:
            return activation_fn(fc)
        return fc

# ====================
# Data Loader 
# ====================
DATA_URL="http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

def _build_vocab(filename):
    with open(filename, "r") as f:
        data =  f.read().replace("\n", "<eos>").split()

    counter = collections.Counter(data)
    # dscending order w.r.t x[1]
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def _to_word_ids(filename, word_to_id):
    with open(filename, "r") as f:
        data =  f.read().replace("\n", "<eos>").split()

    return [word_to_id[word] for word in data if word in word_to_id]

def _download_data(filename, filepath, data_dir):
    def _progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" %(filename,
                                                        float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print("Successfully dowloaded", filename, statinfo.st_size, "bytes.")
    
    with tarfile.open(filepath, mode='r') as tar_:
        print("Openning tarfile...")
        tar_.extractall(path=data_dir)
        os.remove(filepath)


def ptb_raw_data(data_dir, data_name):
    data_dir = os.path.join(data_dir, data_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(os.path.splitext(filepath)[0]):
        _download_data(filename, filepath, data_dir)

    train_path = os.path.join(data_dir, 'simple-examples/data/ptb.train.txt')
    valid_path = os.path.join(data_dir, 'simple-examples/data/ptb.valid.txt')
    test_path  = os.path.join(data_dir, 'simple-examples/data/ptb.test.txt')

    word_to_id = _build_vocab(train_path)
    train_data = _to_word_ids(train_path, word_to_id)
    valid_data = _to_word_ids(valid_path, word_to_id)
    test_data  = _to_word_ids(test_path , word_to_id)

    vocab = len(word_to_id)

    return train_data, valid_data, test_data, vocab

# todo ptb producer
