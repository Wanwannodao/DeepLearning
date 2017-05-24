#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

import utils
from model import PtrNet

class Input():
    def __init__(self, config, enc_data, dec_data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps  = num_steps  = config.num_steps
        self.input_data, self.targets = utils.batch_producer(
            enc_data, dec_data, batch_size, name=name)

class Config():
    batch_size  = 20
    num_steps   = 50
    hidden_size = 256

def main(_):
    config = Config()

    enc_data, dec_data = utils._load_data("./convex_hull_50_train.txt")
    input_ = Input(config, enc_data, dec_data)

    ptrnet = PtrNet(is_training=False,
                    config=config,
                    input_=input_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(sess.run(ptrnet.enc_final_state))
        
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
