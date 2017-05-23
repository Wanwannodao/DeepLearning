#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

_GO  = "output"
_PAD = -1

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
                    
                    # TODO: is this necessary ??
                    while len(dec) != len(enc):
                        dec = np.append(dec, _PAD)

                    enc_in.append(enc)
                    dec_out.append(dec)

            data_len = len(dec_out)
            print("data len %d" % data_len) 

            enc_in  = np.asarray(enc_in, dtype=np.float32)
            enc_in  = np.reshape(enc_in, [data_len, len(enc_in[0]) // 2, 2])
            dec_out = np.asarray(dec_out, dtype=np.int32)
            
            return enc_in, dec_out

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
                             [batch_size, seq_len // 2, 2],
                             [1, 1, 1])
        x.set_shape([batch_size, seq_len // 2, 2 ])

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
    
                    
