#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import utils

class PTBModel():
    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps  = input_.num_steps
        size       = config.hidden_size
        vocab_size = config.vocab_size

        lstm = utils.LSTM(size, is_tuple=True, prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # word embedding
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            # word IDs to embedding mat
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for i in range(num_steps):
                if i > 0: tf.get_variable_scope().reuse_variables()
                out, state = cell(inputs[:, i, :], state)
                outs.append(out)
        outs = tf.stack(outs)
        #s = tf.concat(outs, 1)
        out = tf.reshape(outs, [-1, size])

        logits = utils.fc(out, size, vocab_size, scope="fc")
        loss = -tf.log(logits)
        #loss   = tf.contrib.seq2seq.sequence_loss(
        #    [logits],
        #    [tf.reshape(input_.targets, [-1])],
        #    [tf.ones([batch_size*num_steps], dtype=tf.float32)] )
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr  = tf.Variable(0.0, trainable=False)
        tvars     = tf.trainable_variables()
        grads, _  = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        # learning rate
        self._new_lr    = tf.placeholder(
            tf.float32, shape=[], name="new_lr")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, sess, lr_value):
        sess.run(self._lr_update, feed_dict={self._new_lr: lr_value})
                                
    @property
    def input(self):
        return self._input
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def cost(self):
        return self._cost
    @property
    def final_state(self):
        return self._final_state
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op
