#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import model.StackedLSTM

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "size of batch")
flags.DEFINE_integer("lstm_size", 1, "size of lstm")
flags.DEFINE_integer("num_layers", 1, "# of lstm layers")
flags.DEFINE_integer("num_steps", 1, "# of steps")
FLAGS = flags.FLAGS

def run():

    lstm = LSTM(FLAGS.lstm_size, is_tuple=False)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [lstm] * FLAGS.num_layers,
        state_is_tuple=False)

    initial_state = state = stacked_lstm.zero_state(FLAGS.batch_size, tf.float32)

    for i in range(FLAGS.steps):
        out, state = staeck_lstm(words[:, i], state)

def main(_):
    pass


if __name__ == "__main__":
    tf.app.run()
