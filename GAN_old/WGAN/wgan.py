import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from model import Critic, Generator
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def visualize(generated_images, epoch):
    col, row = 8, 8
    fig, axes = plt.subplots(row, col, sharex=True, sharey=True)
    images = np.squeeze(generated_images, axis=(-1,))*255.5

    for i, array in enumerate(images):
        image = Image.fromarray(array)
        
        ax = axes[int(i/col), int(i%col)]
        ax.axis("off")
        ax.imshow(image)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.2)
    #plt.pause(.01)    
    #plt.show()
    plt.savefig("image_{}.png".format(epoch))

# ====================
# WGAN
# ====================
class WGAN():
    def __init__(self, batch_size):
        self.C = Critic(batch_size)
        self.G = Generator(batch_size)

        self.X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
        self.p = tf.placeholder(tf.float32, name="p")

        self.gen_img = self.G()

        g_logits = self.C(self.gen_img, self.p)

        self.g_loss = -tf.reduce_mean(g_logits)
        self.c_loss = tf.reduce_mean(-self.C(self.X, self.p, reuse=True) + g_logits)
        #self.g_loss = tf.reduce_mean(tf.reduce_sum(g_logits, axis=1))
        #self.c_loss = tf.reduce_mean(tf.reduce_sum(-self.C(self.X, self.p, reuse=True) + g_logits, axis=1))
        
        c_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        g_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        
        c_grads_and_vars = c_opt.compute_gradients(self.c_loss)
        g_grads_and_vars = g_opt.compute_gradients(self.g_loss)
        
        c_grads_and_vars = [[grad, var] for grad, var in c_grads_and_vars \
                            if grad is not None and var.name.startswith("C") ]
        g_grads_and_vars = [[grad, var] for grad, var in g_grads_and_vars \
                            if grad is not None and var.name.startswith("G") ]

        self.c_train_op = c_opt.apply_gradients(c_grads_and_vars)
        self.g_train_op = g_opt.apply_gradients(g_grads_and_vars)

        self.w_clip = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) \
                       for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="C")]
        

if __name__ == "__main__":

    mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_dat", one_hot=True)

    with tf.Session() as sess:        
        wgan = WGAN(batch_size=64)

        summary_writer = tf.summary.FileWriter(DATA_DIR, sess.graph)
        summary_op = tf.summary.merge([tf.summary.scalar("c_loss", -wgan.c_loss)])
        
        sess.run(tf.global_variables_initializer())
        global_step = 0
        for epoch in range(20):
            for step in range(1000):
                
                if (step == 0):
                    visualize(sess.run(wgan.gen_img), epoch)

                for n in range(5):
                    batch_xs, _ = mnist.train.next_batch(64)

                    batch_xs = batch_xs.reshape([-1, 28, 28, 1])

                    feed = {wgan.X: batch_xs, wgan.p: 0.5}
                    _, loss_summary = sess.run([wgan.c_train_op, summary_op],  feed_dict=feed)
                    _ = sess.run(wgan.w_clip)
                    if n == 4 and global_step % 50 == 0: 
                        summary_writer.add_summary(loss_summary, global_step)                

                _ = sess.run(wgan.g_train_op, feed_dict={wgan.p: 0.5})
                global_step += 1
