#!/ust/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import utils
from model import PtrNet

import numpy as np
np.set_printoptions(threshold=np.inf)

class Input():
    def __init__(self, config, enc_data, dec_data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps  = num_steps  = config.num_steps
        self.input_data, self.targets = utils.batch_producer(
            enc_data, dec_data, batch_size, name=name)

class EvalConfig():
    batch_size  = 100
    num_steps   = 50 + 1
    hidden_size = 256


def main(_):
    config             = EvalConfig()
    enc_data, dec_data = utils._load_data("./convex_hull_50_test.txt")

    # an example of plotting convex hull
    utils.plot(enc_data[0], dec_data[0])

    input_ = Input(config, enc_data, dec_data)
    ptrnet = PtrNet(is_training=True,
                    config=config,
                    input_=input_)
    
    with tf.Session() as sess:
        
        
        saver = tf.train.Saver()

        if not os.path.exists("./checkpoint"):
            print("checkpoint is not found")
        else:
            saver.restore(sess, "./checkpoint/model.ckpt")

        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        C = np.asarray(sess.run(ptrnet.C_idx))

        #for i in range(100):
        #    utils.plot(enc_data[i], C[i, :], "eval_{}.png".format(i))
        #print(C)
        idx = [6, 8, 18, 53, 63, 75, 84]
        end = [10, 9, 9, 9, 10, 10, 10]
        print(C[idx])
        for i, j in zip(idx, end):
            utils.plot(enc_data[i], C[i, :j], "eval_{}.png".format(i))
            
        coord.request_stop()
        coord.join(threads)
        



if __name__ == "__main__":
    tf.app.run()
