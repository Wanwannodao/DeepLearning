#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import utils
from utils import convt, leaky_relu, conv, convt, flatten, fc

class Discriminator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def model(self, X, reuse=False):
        
        with tf.variable_scope("D", reuse=reuse):
            conv1 = conv(X,
                         kernel=[5, 5, 3, 64],
                         stride=[1, 2, 2, 1],
                         activation_fn=leaky_relu,
                         scope="conv1")

            conv2 = conv(conv1,
                         kernel=[5, 5, 64, 128],
                         stride=[1, 2, 2, 1],
                         bn=True,
                         activation_fn=leaky_relu,
                         scope="conv2")

            conv3 = conv(conv2,
                         kernel=[5, 5, 128, 256],
                         stride=[1, 2, 2, 1],
                         bn=True,
                         activation_fn=leaky_relu,
                         scope="conv3")
            """
            conv4 = conv(conv3,
                         kernel=[5, 5, 256, 512],
                         stride=[1, 2, 2, 1],
                         bn=True,
                         activation_fn=leaky_relu,
                         scope="conv4")
            """

            flt, dim = flatten(conv3)

            probs = fc(flt, dim, 1, activation_fn=tf.sigmoid, scope="fc1")

            return probs

    def __call__(self, X, reuse=False):
        return self.model(X, reuse=reuse)

class Generator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def model(self, reuse=False):
        with tf.variable_scope("G", reuse=reuse):
            z = tf.random_uniform([self.batch_size, 1024], minval=-1.0, maxval=1.0)
            
            fc1 = fc(z, 1024, 7*7*256, bn=True, activation_fn=tf.nn.relu, scope="fc1")
            fc1 = tf.reshape(fc1, [-1, 7, 7, 256])
            
            convt1 = convt(fc1,
                           kernel=[3, 3, 256, 256],
                           stride=[1, 2, 2, 1],
                           output=[self.batch_size, 14, 14, 256],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt1")
            """
            convt2 = convt(convt1,
                           kernel=[3, 3, 256, 256],
                           stride=[1, 1, 1, 1],
                           output=[self.batch_size, 14, 14, 256],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt2")
            """
            convt3 = convt(convt1,
                           kernel=[3, 3, 256, 256],
                           stride=[1, 2, 2, 1],
                           output=[self.batch_size, 28, 28, 256],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt3")
            """
            convt4 = convt(convt3,
                           kernel=[3, 3, 256, 256],
                           stride=[1, 1, 1, 1],
                           output=[self.batch_size, 28, 28, 256],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt4")
            """
            convt5 = convt(convt3,
                           kernel=[3, 3, 128, 256],
                           stride=[1, 2, 2, 1],
                           output=[self.batch_size, 56, 56, 128],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt5")
            """
            convt6 = convt(convt5,
                           kernel=[3, 3, 64, 128],
                           stride=[1, 1, 1, 1],
                           output=[self.batch_size, 56, 56, 64],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt6")
            convt7 = convt(convt6,
                           kernel=[3, 3, 3, 64],
                           stride=[1, 1, 1, 1],
                           output=[self.batch_size, 56, 56, 3],
                           activation_fn=tf.nn.tanh,
                           scope="convt7")
            
            """
            
            convt6 = convt(convt5,
                           kernel=[3, 3, 64, 128],
                           stride=[1, 2, 2, 1],
                           output=[self.batch_size, 112, 112, 64],
                           bn=True,
                           activation_fn=tf.nn.relu,
                           scope="convt6")
            convt7 = convt(convt6,
                           kernel=[3, 3, 3, 64],
                           stride=[1, 1, 1, 1],
                           output=[self.batch_size, 112, 112, 3],
                           activation_fn=tf.nn.tanh,
                           scope="convt7")
            
            return convt7

    def __call__(self, reuse=False):
        return self.model(reuse=reuse)
        
