#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import utils
from model import PTBModel

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "size of batch")
flags.DEFINE_integer("lstm_size", 1, "size of lstm")
flags.DEFINE_integer("num_layers", 1, "# of lstm layers")
flags.DEFINE_integer("num_steps", 1, "# of steps")
flags.DEFINE_string("data_dir", "./data", "data dir")
flags.DEFINE_string("data_name", "PTB", "data name")
FLAGS = flags.FLAGS

class PTBInput():
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps  = num_steps  = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = utils.ptb_producer(
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
    
def run_epoch(sess, model, eval_op=None):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

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
    raw_data = utils.ptb_raw_data(FLAGS.data_dir, FLAGS.data_name)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=config, data=test_data, name="Test")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
                
            
        with tf.Session() as sess:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(sess, config.lr * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
                train_perplexity = run_epoch(sess, m, eval_op=m.train_op)

                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(sess, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(sess, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
            
if __name__ == "__main__":
    tf.app.run()
