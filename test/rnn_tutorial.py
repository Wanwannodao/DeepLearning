#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import model.StackedLSTM
import reader
import model.PTBModel

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "size of batch")
flags.DEFINE_integer("lstm_size", 1, "size of lstm")
flags.DEFINE_integer("num_layers", 1, "# of lstm layers")
flags.DEFINE_integer("num_steps", 1, "# of steps")
FLAGS = flags.FLAGS

class PTBInput():
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps  = num_steps  = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)
    
class SmallConfig():
    init_scale = 0.1
    lr = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    
def run_epoch(sess, model):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["state"]

        costs += cost
        iters += model.input.num_steps

    return np.exp(costs / iters)

def get_config():
    return SmallConfig()
                          
def main(_):

    # read data
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)

if __name__ == "__main__":
    tf.app.run()
