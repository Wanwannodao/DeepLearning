import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from PIL import Image
import math
import matplotlib.pyplot as plt


DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# helper function
def conv_leaky_relu(input, kernel_shape, stride_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             tf.float32,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=stride_shape,
                        padding="SAME")
    # leaky relu
    conv = conv + biases
    return tf.maximum(0.01*conv, conv)

def conv_transposed(input, kernel_shape, bias_shape, output_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.02))
    biases = tf.get_variable("biases", bias_shape,
                             tf.float32,
                             initializer=tf.constant_initializer(0.0))
    convt = tf.nn.conv2d_transpose(input, weights,
                                   strides=[1, 2, 2, 1],
                                   output_shape=output_shape, padding="SAME")

    return convt + biases
                            

def fully_connected(input, in_dim, out_dim):
    W = tf.get_variable("W", [in_dim, out_dim], tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [out_dim], tf.float32,
                        initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(input, W), b)

def flatten(input):
    shape = input.get_shape()[1:].as_list()
    dim = np.prod(shape)
    return tf.reshape(input, [-1,dim]), dim

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

# Generator
class Generator():
    def __init__(self, batch_size=32, reuse=False, trainable=True):
        self.p = tf.placeholder(tf.float32)
        
        with tf.variable_scope("G", reuse=reuse):        
            self.out_img = self.build_G_model(batch_size, reuse)
        
        with tf.variable_scope("D", reuse=True):
            probs = Discriminator.build_D_model(self.out_img, self.p, batch_size, True)
            
            #entropy = tf.reduce_sum(tf.ones(batch_size, tf.float32) - tf.log(self.probs),
            #                               axis=1)
            entropy = -tf.reduce_sum(tf.log(probs), axis=1)
            self.loss = tf.reduce_mean(entropy)

        with tf.variable_scope("G", reuse=reuse):
            if trainable:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars \
                                       if grad is not None and var.name.startswith("G")]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
                                                               
    def build_G_model(self, batch_size, reuse):

        input = tf.random_uniform([batch_size, 100], minval=-1.0, maxval=1.0)

        with tf.variable_scope("fc1", reuse=reuse):
            fc1 = fully_connected(input, 100, 1024)
            batch_mean1, batch_variance1 = tf.nn.moments(fc1, axes=[0])
            bn1 = tf.nn.batch_normalization(fc1, batch_mean1, batch_variance1,
                                            None, None, 1e-5)
            relu1 = tf.nn.relu(bn1)

        with tf.variable_scope("fc2", reuse=reuse):
            fc2 = fully_connected(relu1, 1024, 128*7*7)
            batch_mean2, batch_variance2 = tf.nn.moments(fc2, axes=[0])
            bn2 = tf.nn.batch_normalization(fc2, batch_mean2, batch_variance2,
                                            None, None, 1e-5)
            relu2 = tf.nn.relu(bn2)



        img = tf.reshape(relu2, [-1, 7, 7, 128])

        with tf.variable_scope("convt1", reuse=reuse):
            convt1 = conv_transposed(img, kernel_shape=[5, 5, 64, 128],
                                     bias_shape=[64],
                                     output_shape=[batch_size, 14, 14, 64])
            batch_mean3, batch_variance3 = tf.nn.moments(convt1, axes=[0, 1, 2])
            bn3 = tf.nn.batch_normalization(convt1, batch_mean3, batch_variance3,
                                            None, None, 1e-5)
            relu3 = tf.nn.relu(bn3)

        with tf.variable_scope("convt2", reuse=reuse):
            convt2 = conv_transposed(relu3, kernel_shape=[5, 5, 1, 64],
                                     bias_shape=[1],
                                     output_shape=[batch_size, 28, 28, 1])
            return tf.nn.tanh(convt2)

    def generate_images(self):
        return self.out_img


# Discriminator
class Discriminator():
    def __init__(self, batch_size=32, reuse=False, trainable=True):
        self.X = tf.placeholder(shape=[None, 28, 28, 1],
                           dtype=tf.float32,
                           name="X")

        self.y = tf.placeholder(shape=[None, 1],
                           dtype=tf.float32,
                           name="y")

        self.p = tf.placeholder(tf.float32, name="p")


        with tf.variable_scope("D", reuse=reuse):
                
            self.probs = Discriminator.build_D_model(self.X, self.p, batch_size, reuse)
            self.entropy = -tf.reduce_sum(tf.log(self.probs[0:64]) + tf.log(tf.ones(64, tf.float32)-self.probs[64:batch_size]),
                                          axis=1, name="entropy")
            self.loss = tf.reduce_mean(self.entropy, name="loss")
            
            if trainable:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.1)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    @classmethod
    def build_D_model(self, X, p, batch_size, reuse):

        with tf.variable_scope("conv1", reuse=reuse):
            conv1 = conv_leaky_relu(X, [5, 5, 1, 64], [1, 2, 2, 1], [64])
        with tf.variable_scope("conv2", reuse=reuse):
            conv2 = conv_leaky_relu(conv1, [5, 5, 64, 128], [1, 2, 2, 1], [128])


        flt, dim = flatten(conv2)
        
        with tf.variable_scope("fc1", reuse=reuse):
            fc1 = fully_connected(flt, dim , 256)
            fc1 = tf.maximum(0.01*fc1, fc1)
            
        dropout = tf.nn.dropout(fc1, p)

        with tf.variable_scope("fc2", reuse=reuse):
            fc2 = fully_connected(dropout, 256, 1) # logits
        return tf.nn.sigmoid(fc2)
    
if __name__ == "__main__":

    mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_dat", one_hot=True)
    
    with tf.Session() as sess:
        D = Discriminator(batch_size=128)
        G = Generator(batch_size=64)

        summary_writer = tf.summary.FileWriter(DATA_DIR, sess.graph)

        D_summary_op = tf.summary.merge([tf.summary.scalar("D_loss", D.loss)])
        G_summary_op = tf.summary.merge([tf.summary.scalar("G_loss", G.loss)])

        sess.run(tf.global_variables_initializer())        
        
        for epoch in range(30):

            for step in range(2000):

                batch_xs, _ = mnist.train.next_batch(64)
                               
                imgs = sess.run(G.generate_images())
                batch_xs = batch_xs.reshape([-1, 28, 28, 1])
                batch_xs = np.concatenate((batch_xs, imgs), axis=0)

                if step  == 0:
                    visualize(imgs, epoch)

                batch_ys = np.asarray([[1]]*64 + [[0]]*64, dtype=np.float32)

                feed_D = {D.X: batch_xs, D.y: batch_ys, D.p: 0.5}
                feed_G = {G.p: 0.5}

                _, d_loss, D_summary = sess.run([D.train_op, D.loss, D_summary_op], feed_D)
                _, g_loss, G_summary = sess.run([G.train_op, G.loss, G_summary_op], feed_G)

                summary_writer.add_summary(D_summary, step)
                summary_writer.add_summary(G_summary, step)

            print("epoch:{}, d_loss:{}, g_loss{}".format(epoch, d_loss, g_loss))

        imgs = sess.run(G.generate_images())
        visualize(imgs, 31)
