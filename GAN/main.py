import os
import tensorflow as tf
from model import DCGAN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "# epochs")
flags.DEFINE_integer("batch_size", 32, "size of batch")
flags.DEFINE_integer("step", 2000, "# steps")
flags.DEFINE_bool("log_dev_placement", False, "placement logs")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of checkpoint dir")
flags.DEFINE_bool("reset", False, "reset learning")
flags.DEFINE_string("data_dir", "data", "name of data dir")
flags.DEFINE_string("data", "anime", "name of data")
FLAGS = flags.FLAGS

DATA_DIR="./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def main(_):
    with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_dev_placement)) as sess:
        dcgan = DCGAN(sess, batch_size=FLAGS.batch_size,
                      #in_dim=[28,28,1], z_dim=100)
                      in_dim=[112,112,3], z_dim=100)
        dcgan.train(FLAGS)

if __name__ == "__main__":
    tf.app.run()
