#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from model import Discriminator, Generator

class LSGAN:
    def __init__(self, input_shape):
        self.batch_size = input_shape[0]

        self.D = Discriminator(self.batch_size)
        self.G = Generator(self.batch_size)

        
        self.X = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        
        self.gen_img = self.G()
        
        self.g_loss = 0.5*(tf.reduce_mean( (self.D(self.G(reuse=True)) - 1.0)**2 ))
        self.d_loss = 0.5*(tf.reduce_mean((self.D(self.X, reuse=True) - 1.0)**2 )\
                           + tf.reduce_mean( (self.D( self.G(reuse=True), reuse=True))**2 ) )

        g_opt = tf.train.AdamOptimizer(learning_rate=4e-3,beta1=0.5)
        d_opt = tf.train.AdamOptimizer(learning_rate=1e-3,beta1=0.5)

        g_grads_and_vars = g_opt.compute_gradients(self.g_loss)
        d_grads_and_vars = d_opt.compute_gradients(self.d_loss)

        g_grads_and_vars = [[grad, var] for grad, var in g_grads_and_vars \
                            if grad is not None and var.name.startswith("G")]
        d_grads_and_vars = [[grad, var] for grad, var in d_grads_and_vars \
                            if grad is not None and var.name.startswith("D")]

        self.g_train_op = g_opt.apply_gradients(g_grads_and_vars)
        self.d_train_op = d_opt.apply_gradients(d_grads_and_vars)
        
        
