import tensorflow as tf
import numpy as np

# ====================
# Helper functions
# ====================
def leaky_relu(x):
    return tf.maximum(0.01*x, x)

def conv(x, kernel, stride, padding="SAME", activation_fn=None, scope="conv", bias=None):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", kernel, dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(x, W, strides=stride, padding="SAME")
        
        if bias is not None:
            b = tf.get_variable("b", bias, dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
            conv = conv + b

        if activation_fn is not None:
            return activation_fn(conv)
        return conv

def convt(x, kernel, stride, output, padding="SAME", activation_fn=None, scope="convt", bias=None):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", kernel, dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        convt = tf.nn.conv2d_transpose(x, W, strides=stride, output_shape=output, padding=padding)
        if bias is not None:
            b = tf.get_variable("b", bias, dtype=tf.float32,
                                initialiezr=tf.constant_initializer(0.0))

        if activation_fn is not None:
            return activation_fn(convt)
        return convt

def fully_connected(x, in_dim, out_dim, activation_fn=None, scope="fc"):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", [in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, W), b)
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
# Generator
# ====================
class Generator():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def build_model(self, reuse):
        with tf.variable_scope("G", reuse=reuse):
            z = tf.random_uniform([self.batch_size, 100], minval=-1.0, maxval=1.0)

            fc1 = tf.nn.relu(batch_norm(fully_connected(z, 100, 1024, scope="fc1"), axes=[0]))

            fc2 = tf.nn.relu(batch_norm(fully_connected(fc1, 1024, 128*7*7, scope="fc2"), axes=[0]))
            fc2 = tf.reshape(fc2, [-1, 7, 7, 128])

            convt1 = tf.nn.relu(batch_norm(convt(fc2, kernel=[5, 5, 64, 128],
                                                 stride=[1, 2, 2, 1],
                                                 output=[self.batch_size, 14, 14, 64],
                                                 scope="convt1"), axes=[0, 1, 2]))
            convt2 = convt(convt1, kernel=[5, 5, 1, 64],
                           stride=[1, 2, 2, 1],
                           output=[self.batch_size, 28, 28, 1],
                           activation_fn=tf.nn.tanh,
                           scope="convt2")
            return convt2

    def __call__(self, reuse=False):
        return self.build_model(reuse)

# ====================
# Critic
# ====================
class Critic():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def build_model(self, X, p, reuse=False):
        with tf.variable_scope("C", reuse=reuse):
            conv1 = conv(X, kernel=[5, 5, 1, 64], stride=[1, 2, 2, 1],
                         activation_fn=leaky_relu, scope="conv1")
            conv2 = conv(conv1, kernel=[5, 5, 64, 128], stride=[1, 2, 2, 1],
                         activation_fn=leaky_relu, scope="conv2")

            convt2, dim = flatten(conv2)
            fc1 = fully_connected(convt2, dim, 256, activation_fn=leaky_relu, scope="fc1")

            #dropout = tf.nn.dropout(fc1, p)

            # without sigmoid
            logits = fully_connected(fc1, 256, 1, scope="fc2")
            return logits

    def __call__(self, X, p, reuse=False):
        return self.build_model(X, p, reuse)
