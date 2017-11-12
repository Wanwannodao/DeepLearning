#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import urllib.request
import zipfile
import six.moves.cPickle as pickle
import numpy as np

import logging
import cv2

from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

# ====================
# Helper functions
# ====================
def leaky_relu(x):
    return tf.maximum(0.01*x, x)

def conv(x, kernel, stride, padding="SAME", fmt="NHWC", bn=False, activation_fn=None, scope="conv", bias=None):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", kernel, dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(x, W, strides=stride, data_format=fmt, padding="SAME")
        
        if bias is not None:
            b = tf.get_variable("b", bias, dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
            conv = conv + b

        if bn:
            conv = batch_norm(conv, axes=[0])

        if activation_fn is not None:
            return activation_fn(conv)
        return conv

def convt(x, kernel, stride, output, padding="SAME", fmt="NHWC", bn=True, activation_fn=None, scope="convt", bias=None):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", kernel, dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        convt = tf.nn.conv2d_transpose(x, W, strides=stride, output_shape=output, padding=padding,
                                       data_format=fmt)
        if bias is not None:
            b = tf.get_variable("b", bias, dtype=tf.float32,
                                initialiezr=tf.constant_initializer(0.0))

        if bn:
            convt = batch_norm(convt, axes=[0, 1, 2])
            
        if activation_fn is not None:
            return activation_fn(convt)
        return convt

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

def flatten(x):
    shape = x.get_shape()[1:].as_list()
    dim = np.prod(shape)
    return tf.reshape(x, [-1, dim]), dim

def batch_norm(x, axes):
    mean, var = tf.nn.moments(x, axes=axes)
    return tf.nn.batch_normalization(x, mean, var, None, None, 1e-5)

# ====================
# visualize
# ====================
def visualize(generated_images, epoch):
    col, row = 8, 8
    fig, axes = plt.subplots(row, col, sharex=True, sharey=True)
    #images = np.squeeze(generated_images, axis=(-1,))*127.5
    images = generated_images*127.5 + 127.5
    #images = generated_images*255.0
    for i, array in enumerate(images):

        image = Image.fromarray(array.astype(np.uint8))

        ax = axes[int(i/col), int(i%col)]
        ax.axis("off")
        ax.imshow(image)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2)
    #plt.pause(.01)    
    #plt.show()
    plt.savefig("image_{}.png".format(epoch))

# ====================
# Data Loader 
# ====================
DATA_URL="http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip"
""" from https://github.com/nagadomi/lbpcascade_animeface """
CASCADE_PATH = './lbpcascade_animeface.xml'

class Loader:
    def __init__(self, data_dir, data_name, batch_size):
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.cur = 0
        
        self._load_data(data_dir, data_name)
        self.batch_num = int(len(self.data) / batch_size)
        self.data_num = len(self.data)        

    def reset(self):
        np.random.shuffle(self.data)
        self.cur = 0
        
    def next_batch(self):
        if self.cur + self.batch_size > self.data_num:
            return None

        batch = self.data[self.cur:self.cur+self.batch_size]
        self.cur += self.batch_size
        return batch

    def _download_data(self, filename, filepath, data_dir):
    
        """ Download data and unzip """
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %(filename,
                                                            float(count * block_size) / float(total_size) *100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes.")

        with zipfile.ZipFile(filepath, 'r') as zip_:
            print("Unzipping...")
            zip_.extractall(path=data_dir)
            os.remove(filepath)

    def _load_data(self, data_dir, data_name):
    
        data_dir = os.path.join(data_dir, data_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            _download_data(filename, filepath, data_dir)

        pkl_path = os.path.join(data_dir, "data.pkl")

        if os.path.exists(pkl_path):
            print("Loading from pkl...")
            self.data = pickle.load(open(pkl_path, "rb"))
            self.data_num = len(self.data)
        else:
            self.data_num = 0
            data_dir = os.path.join(data_dir, filename.split('.')[0], "thumb")
            dirs = os.listdir(data_dir)
            data = []
            for i, d in enumerate(dirs):
                files = os.listdir(os.path.join(data_dir, d))
        
                sys.stdout.write("\rDirectories: {}/{}".format(i, len(dirs)))
                sys.stdout.flush()
        
                for f in files:
                    root, ext = os.path.splitext(f)
                    if ext == ".png":
                        # BGR
                        img = cv2.imread(os.path.join(data_dir, d, f))
                        # BGR2RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        gray = cv2.equalizeHist(gray)
                        cascade = cv2.CascadeClassifier(CASCADE_PATH)
                        face = cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=2,
                            minSize=(10, 10))
                
                        #print("{}".format(face))
                        if len(face) == 1:
                            x, y, w, h = face[0]
                            img = img[y:y+h, x:x+w]
                            img = cv2.resize(img, (112, 112))
                            
                            data.append(img)
                            #data.append(img.transpose(2, 0, 1))
                            #print("{}".format(img.shape))
                            #cv2.namedWindow('window')
                            #cv2.imshow('window', img)
                            #cv2.waitKey(0)
                            #cv2.destroyAllWindows()
                            self.data_num += 1

            self.data = np.asarray(data, dtype=np.float32)
            pickle.dump(data, open(pkl_path, "wb"), -1)
