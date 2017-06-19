#!/usr/bin/env python33
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import inspect
import matplotlib.pyplot as plt

_GO  = "output"
_STOP = 0

# ====================
# ops
# ====================

# basic lstm cell (with dropout layer)
def LSTM(size, is_tuple=True, init=None, prob=1.0):
    cell = tf.contrib.rnn.LSTMCell(num_units=size,
                                   use_peepholes=False,
                                   initializer=init,
                                   state_is_tuple=is_tuple,
                                   reuse=tf.get_variable_scope().reuse)
                                   
    if prob < 1.0:
        return tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=prob)

    return cell

# batch normalization
def batch_norm(x, axes):
    mean, var = tf.nn.moments(x, axes=axes)
    return tf.nn.batch_normalization(x, mean, var, None, None, 1e-5)

# fully connected layer
def fc(x, in_dim, out_dim, init_w=None, init_b=None, bn=False, a_fn=None, scope="fc"):
    if init_w is None:
        init_w = tf.random_noramal_initializer(stddev=0.02)
    if init_b is None:
        init_b = tf.constant_initializer(0.0)
        
    with tf.variable_scope(scope):
        W  = tf.get_variable("W", [in_dim, out_dim], dtype=tf.float32,
                             initializer=init_w)
        b  = tf.get_variable("b", [out_dim], dtype=tf.float32,
                             initializer=init_b)
        fc = tf.nn.bias_add(tf.matmul(x, W), b)

        if bn:
            fc = batch_norm(fc, axes=[0])

        if a_fn is not None:
            return a_fn(fc)

        return fc
    
# ====================
# data loder
# ====================

def _load_data(data_path):
    enc_in  = []
    dec_out = []
    
    if tf.gfile.Exists(data_path):
        with tf.gfile.GFile(data_path, mode="r") as f:

            for line in f:
                line = line.strip().split(_GO)
                                                                      
                if len(line) == 2:
                    enc = line[0].strip().split()
                    dec = line[1].strip().split()

                    # based on fig.1 (b)
                    enc.insert(0, '-1.0')
                    enc.insert(0, '-1.0')

                    while len(dec) != len(enc) // 2:
                        dec = np.append(dec, _STOP)

                    enc_in.append(enc)
                    dec_out.append(dec)

            data_len = len(dec_out)
            print("data len %d" % data_len) 

            enc_in  = np.asarray(enc_in, dtype=np.float32)
            enc_in  = np.reshape(enc_in, [data_len, len(enc_in[0]) // 2, 2])
            dec_out = np.asarray(dec_out, dtype=np.int32)
            
            return enc_in, dec_out

# ====================
# batch generator
# ====================

def batch_producer(enc, dec, batch_size, name=None):
    data_len   = enc.shape[0]
    seq_len    = enc.shape[1]
    epoch_size = data_len // batch_size

    print("epoch size: %d " % epoch_size)
    
    with tf.name_scope(name, "batch", [enc, dec, batch_size]):
        enc = tf.convert_to_tensor(enc, name="enc", dtype=tf.float32)
        dec = tf.convert_to_tensor(dec, name="dec", dtype=tf.int32) 

        # generator 
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

        x = tf.strided_slice(enc, [0, 0, 0],
                             [batch_size, seq_len, 2],
                             [1, 1, 1])
        x.set_shape([batch_size, seq_len, 2 ])

        y = tf.strided_slice(dec, [0, 0],
                             [batch_size, seq_len],
                             [1, 1])
                        
        y.set_shape([batch_size, seq_len])
        
        return x, y

# for test

#if __name__ == "__main__":
#    enc_in, dec_out = _load_data("./convex_hull_50_train.txt")
#    print(enc_in.shape)
#    print(dec_out.shape)
#    #print(enc_in)
#    x_batch, y_batch = batch_producer(enc_in, dec_out, batch_size=20)
               
#    with tf.Session() as sess:
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#        print(sess.run([x_batch, y_batch]))

#        coord.request_stop()
#        coord.join(threads)
    
                    
# ====================
# visualization
# ====================
def visualize_loss_from_file(filename):
    loss = np.loadtxt(filename, delimiter=",")
    plt.plot(loss, color="orange")
    plt.title("Learning curve")
    plt.xlabel("epoch", fontname="serif")
    plt.ylabel("loss", fontname="serif")

    plt.savefig("loss.png")

def plot(data, dec, filename="data.png"):
    idx    = dec[ np.where(dec != 51)[0] ]
    convex = data[idx, :] 
    
    x = data[1:, 0]
    y = data[1:, 1]
    convex_x = convex[1:, 0]
    convex_y = convex[1:, 1]
    
    plt.scatter(x, y)
    plt.plot(convex_x, convex_y, color="orange")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.savefig(filename)

    plt.clf()
    plt.close()
