import tensorflow as tf
from model import QFunction
from pr import PrioritizedReplayBuf
import collections
import numpy as np
import random


def copy_params(src, dst):
    src_params = [v for v in tf.trainable_variables() if v.name.startswith(src.scope)]
    dst_params = [v for v in tf.trainable_variables() if v.name.startswith(dst.scope)]

    op = [d.assign(s) for s, d in zip(src_params, dst_params)]

    return op

def Hurber_loss(x, y, delta=1.0):
    error = tf.abs(y - x)
    cond = tf.less(error, delta)
    return tf.where(cond,
                    0.5*tf.square(error),
                    delta*error - 0.5*tf.square(delta))

class DDQN_PR:
    def __init__(self, input_shape, action_n, N, alpha=0.5, beta=0.5, beta_decay=50000, gamma=0.99):
        self.shape = input_shape
        self.batch_size = input_shape[0]

        # Prioritized Replay Memory
        self.pr = PrioritizedReplayBuf(N=N,
                                       alpha=alpha,
                                       beta=beta, beta_decay=beta_decay,
                                       batch_size=self.batch_size)
        # Importance Sampling weights
        self.is_w = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32)
        
        Q = QFunction(input_shape, action_n, scope="Q")
        target_Q = QFunction(input_shape, action_n, scope="target_Q")

        # Forward Q
        self.s = tf.placeholder(shape=[None]+input_shape[1:], dtype=tf.float32)
        self.a = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32)
        self.probs = Q(self.s, s_bias=False)

        # add offset 
        first = tf.expand_dims(tf.range(self.batch_size), axis=1)
        indices = tf.concat(values=[first, self.a], concat_dim=1)
        # gather corresiponding q_vals
        self.q_val = tf.expand_dims(tf.gather_nd(self.probs, indices), axis=1)

        # TD target
        self.done = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32)
        self.r = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.float32)
        self.s_ = tf.placeholder(shape=input_shape, dtype=tf.float32)

        # D-DQN
        a_max = tf.expand_dims(tf.argmax(Q(self.s_, reuse=True), axis=1), axis=1)
        a_max = tf.to_int32(a_max)
        target_q_val = tf.expand_dims(
            tf.gather_nd(target_Q(self.s_),
                         tf.concat(values=[first, a_max], concat_dim=1))
            , axis=1)
        self.y = self.r + gamma*(1.0 - self.done)*target_q_val
        # Error Clipping
        # TD-error
        self.delta = Hurber_loss(self.q_val, self.y)
        # Importance sampling
        max_is = tf.reduce_max(self.is_w)
        self.loss = tf.reduce_mean( (self.is_w / max_is)  * self.delta)
        

        # Update Q
        # reducing step-size by a factor of four
        opt = tf.train.RMSPropOptimizer(0.00025/4, 0.99, 0.0, 1e-6)
        grads_and_vars = opt.compute_gradients(self.loss)
        grads_and_vars = [[grad, var] for grad, var in grads_and_vars \
                          if grad is not None and (var.name.startswith("Q") or var.name.startswith("shared"))]
        self.train_op = opt.apply_gradients(grads_and_vars)
        
        # Update target Q
        self.target_train_op = copy_params(Q, target_Q)
        
        
    def update_target(self, sess):
        _ = sess.run(self.target_train_op)
            
    def update(self, sess):
        # sample from replay buffer
        d, is_w = self.pr.stratified_sample()
        is_w = np.expand_dims(is_w, axis=1)
        samples = [ e['transition']for e in d]
    
        s = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        a = np.asarray([[sample[1]] for sample in samples], dtype=np.int32)
        r = np.asarray([[sample[2]] for sample in samples], dtype=np.float32)
        done = np.asarray([[sample[3]] for sample in samples], dtype=np.float32)
        s_ = np.asarray([sample[4] for sample in samples], dtype=np.float32)

        feed={self.s:s, self.a:a, self.r:r, self.done:done, self.s_:s_, self.is_w:is_w}
        _, delta = sess.run([self.train_op, self.delta], feed_dict=feed)

        self.pr.update_delta(d, delta)

    def greedy(self, s, sess):
        probs = sess.run( self.probs, feed_dict={self.s:s})
        return np.argmax(probs)
    
    def set_exp(self, exp, init=False):
        self.pr.insert(exp, init)

    
