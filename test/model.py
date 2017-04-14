#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import utils

class StackedLSTM:
    def __init__():
        pass

    def _model(self, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            lstm = utils.lstm(size, is_tuple=False)
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers,
                                                       state_is_tuple=False)

    def __call__(self):
        return self._model()
            
