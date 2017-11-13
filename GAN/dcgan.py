from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from utils import visualize, Model, plot, save_gif, Loader

def leaky_relu(input, x=0.01):
    return tf.maximum(x*input, input)

class DCGAN(Model):

    def __init__ (self, sess, batch_size=32,
                  in_dim=[28,28,1], z_dim=100):
        self.sess       = sess
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.z_dim = z_dim

        self.X = X = tf.placeholder(shape=[batch_size]+in_dim,
                                dtype=tf.float32,
                                name="Real")
        self.z = tf.placeholder(shape=[None, z_dim],
                                dtype=tf.float32,
                                name="z")

        self.G  = self.generator(self.z)
        self.D, logits  = self.discriminator(self.X, reuse=False)
        self.D_, logits_ = self.discriminator(self.G, reuse=True)
        self.sampler = self.generator(self.z, reuse=True, is_training=False)

        d_loss_g = tf.log(tf.ones_like(self.D_)) - self.D_
        d_loss_x = tf.log(self.D)
        self.d_loss = -tf.reduce_mean(d_loss_g + d_loss_x) # max
        self.g_loss = tf.reduce_mean(d_loss_g)             # min

        self.d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.1)
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.1)

        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="D") # batch_norm
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="G")
        with tf.control_dependencies(d_update_ops):
            d_grads_and_vars = self.d_optimizer.compute_gradients(self.d_loss)
            self.d_train_op = self.d_optimizer.apply_gradients(
                [[grad, var] for grad, var in d_grads_and_vars if grad is not None and var.name.startswith("D")])
        with tf.control_dependencies(g_update_ops):
            g_grads_and_vars = self.g_optimizer.compute_gradients(self.g_loss)
            self.g_train_op = self.g_optimizer.apply_gradients(
                [[grad, var] for grad, var in g_grads_and_vars if grad is not None and var.name.startswith("G")])

        # saver
        self.saver = tf.train.Saver()
        
    def discriminator(self, X, y=None, reuse=False, is_training=True):
        batch_norm_params = {"is_training":is_training, "trainable":True}
        # Descriminator
        with tf.variable_scope("D", reuse=reuse):
            h0 = tf.contrib.layers.conv2d(X, 64, kernel_size=5, stride=2,
                                          activation_fn=leaky_relu,
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=batch_norm_params,
                                          scope="h0")
            h1 = tf.contrib.layers.conv2d(h0, 128, kernel_size=5, stride=2,
                                          activation_fn=leaky_relu,
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=batch_norm_params,
                                          scope="h1")
            
            h2 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(h1), 256, activation_fn=leaky_relu, scope="h2")
            
            h2 = tf.contrib.layers.dropout(h2, keep_prob=0.5, is_training=is_training, scope="dropout")

            logits = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(h2), 1, activation_fn=None, scope="logits")
            probs = tf.nn.sigmoid(logits)

            return probs, logits
            
    def generator(self, z, y=None, reuse=False, is_training=True):
        batch_norm_params = {"is_training":is_training, "trainable":True}
        # Generator
        with tf.variable_scope("G", reuse=reuse):
            h0 = tf.contrib.layers.fully_connected(z, 1024, activation_fn=tf.nn.relu,
                                                   normalizer_fn=tf.contrib.layers.batch_norm,
                                                   normalizer_params=batch_norm_params,
                                                   scope="h0")
            height  = self.in_dim[0]//4
            width   = self.in_dim[1]//4
            channel = self.in_dim[2]
            h1 = tf.contrib.layers.fully_connected(h0, 128*height*width, activation_fn=tf.nn.relu,
                                                   normalizer_fn=tf.contrib.layers.batch_norm,
                                                   normalizer_params=batch_norm_params,
                                                   scope="h1")
            h2 = tf.contrib.layers.conv2d_transpose( tf.reshape(h1, shape=[-1,height,width,128]),
                                                     num_outputs=64,
                                                     kernel_size=5, stride=2,
                                                     activation_fn=tf.nn.relu,
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params=batch_norm_params,
                                                     scope="h2")
            return tf.contrib.layers.conv2d_transpose(h2, num_outputs=channel,
                                                      kernel_size=5, stride=2,
                                                      activation_fn=tf.nn.tanh,
                                                      scope="img")
    def train(self, config=None):
        #mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_dat", one_hot=True)

        loader = Loader(config.data_dir, config.data, config.batch_size)

        loaded = False
        if not config.reset:
            loaded, global_step = self.restore(config.checkpoint_dir)
        if not loaded:
            tf.global_variables_initializer().run()
            global_step = 0

        d_losses = []
        g_losses = []
        steps = []
        gif = []
        for epoch in range(config.epoch):
            loader.reset()
            #for idx in range(config.step):
            for idx in range(loader.batch_num):
                #batch_X, _ = mnist.train.next_batch(config.batch_size)
                #batch_X = batch_X.reshape([-1]+self.in_dim)
                batch_X = np.asarray(loader.next_batch(), dtype=np.float32)
                #batch_X = (batch_X*255.-127.5)/127.5
                batch_X = (batch_X - 127.5)/127.5
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim])
                
                _, d_loss = self.sess.run([self.d_train_op, self.d_loss],
                              feed_dict={self.X: batch_X, self.z: batch_z})
                _, g_loss = self.sess.run([self.g_train_op, self.g_loss],
                              feed_dict={self.z: batch_z})
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                steps.append(global_step)
                global_step += 1
                
            print(" [Epoch {}] d_loss:{}, g_loss:{}".format(epoch, d_loss, g_loss))
            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim])
            imgs = self.sess.run(self.sampler, feed_dict={self.z: batch_z})
            gif.append(visualize(imgs, epoch, config.data))
            self.save("{}_{}".format(config.checkpoint_dir, config.data), global_step, model_name="dcgan")

        plot({'d_loss':d_losses, 'g_loss':g_losses}, steps, title="DCGAN loss ({})".format(config.data), x_label="Step", y_label="Loss")
        save_gif(gif, "gen_img_{}".format(config.data))
