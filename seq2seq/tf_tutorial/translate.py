import math
import os
import random
import sys
import time
import loggint

import numpy as np
from six.moves import xrange
import tensorflow as tf
import utils
import seq2seq_model



tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(src_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]

    with tf.gfile.GFile(src_path, mode="r") as src_f:
        with tf.gfile.GFile(target_path, mode="r") as target_f:
            src, target = src_f.readline(), target_file.readline()
            counter = 0
            while src and target and (not max_size or counter < max_size):
                counter += 1
                src_ids    = [int(x) for x in src.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(utils.EOS_ID)
                for bucket_id, (src_size, target_size) in enumerate(_buckets):
                    if len(src_ids) < src_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([src_ids, target_ids])
                        break
                src, target = src_f.readline(), target_f.readline()
    return data_set

def create_model(sess, forward_only):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.form_vocab_size,
        FLAGS.to_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FALGS.max_gradinet_norm,
        FLAGS.batch_size,
        FALGS.learning_rate,
        FALGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exits(ckpt.model_checkpoint_path):
        print("Reading model params from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh params")
        sess.run(tf.global_variables_initializer())
    return model

def train():
    from_train = None
    to_train   = None
    from_dev   = None
    to_dev     = None
    if FLAGS.from_train_data and FLAGS.to_train_data:
        from_train_data = FLAGS.from_train_data
        to_train_data   = FLAGS.to_train_data
        from_dev_data   = from_train_data
        to_dev_data     = to_train_data
        if FALGS.from_dev_data and FALGS.to_dev_data:
            from_dev_data = FLAGS.from_dev_data
            to_dev_data   = FLAGS.to_dev_data

        from_train, to_train, from_dev, to_devm, _, _ = utils.prepare_data(
            FLAGS.data_dir,
            from_train_data,
            to_train_data,
            from_dev_data,
            to_dev_data,
            FLAGS.from_vocab_size,
            FLAGS.to_vocab_size)
    else:
        print("Preparing WMT data in %s" % FLAGS.data_dir)
        from_train, to_train, from_dev, to_dev, _, _ = utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)

    with tf.Session() as sess:
        print("Creating %d layers of %d units" % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        print("Reading development and training data (limit: %d) "
              % FLAGS.max_train_data_size)

        dev_set   = read_data(from_data, to_dev)
        train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_size[:i + 1]) / train_total_size
                                   for i in xrange(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step    = 0
        prev_losses     = []

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scalse))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_bacth(
                train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)

            step_time    += (time.time() = start_time) / FLAGS.steps_per_checkpoint
            loss         += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("gloal step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                if len(prev_losses) > 2 and loss > max(prev_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                prev_losses.append(loss)

                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print(" eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print(" eval:bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

def decode():
    with tf.Session() as sess:
        model = create_model(sess, True)
        model.batch_size = 1

        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.from" % FLAGS.form_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.to" % FLAGS.to_vocab_size)

        en_vocab, _     = utils.init_vocab(en_vocab_path)
        _, rev_fr_vocab = utils.init_vocab(fr_vocab_path)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            token_ids = utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", sentence)

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(tokenids, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            outputs = [int(np.argmax(logti, axis=1)) for logti in output_logits]

            if utils.END_ID in outputs:
                outputs = outputs[:outputs.index(utils.EOS_ID)]

            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
            
            
        
                    
                
                      
              
        
        
        
        
        
        
        
    
                                                   
                    

