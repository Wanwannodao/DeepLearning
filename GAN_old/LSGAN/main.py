#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import utils
from utils import Loader
from lsgan import LSGAN
import numpy as np
import tensorflow as tf

import cv2

tf.app.flags.DEFINE_integer("batch_size", 64, "size of mini-batch")
tf.app.flags.DEFINE_string("data", "anime", "data name")
tf.app.flags.DEFINE_string("data_dir", "data", "data dir")
tf.app.flags.DEFINE_integer("d", 1, "# of D update iters")

FLAGS = tf.app.flags.FLAGS

def main(_):
    loader = Loader(FLAGS.data_dir, FLAGS.data, FLAGS.batch_size)
    print("# of data: {}".format(loader.data_num))
    with tf.Session() as sess:                                
        lsgan = LSGAN([FLAGS.batch_size, 112, 112, 3])
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(10000):
            loader.reset()

            for step in range(int(loader.batch_num/FLAGS.d)):
                if (step == 0 and epoch % 1 == 100):
                    utils.visualize(sess.run(lsgan.gen_img), epoch)
                    
                for _ in range(FLAGS.d):
                    batch = np.asarray(loader.next_batch(), dtype=np.float32)
                    batch = (batch-127.5) / 127.5
                    #print("{}".format(batch.shape))
                    feed={lsgan.X: batch}
                    _ = sess.run(lsgan.d_train_op, feed_dict=feed)
                        #utils.visualize(batch, (epoch+1)*100)
                
                #cv2.namedWindow("window")
                #cv2.imshow("window", cv2.cvtColor(batch[0], cv2.COLOR_RGB2BGR))
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                    
                _ = sess.run(lsgan.g_train_op)
             

if __name__ == "__main__":
    tf.app.run()
