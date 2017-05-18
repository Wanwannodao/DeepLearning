import tensorflow as tf
import numpy as np
_GO  = "output"
_PAD = -1


def _load_data(data_path):
    enc_in  = []
    dec_out = []
    
    if tf.gfile.Exists(data_path):
        with tf.gfile.GFile(data_path, mode="r") as f:
            for line in f:
                line = line.strip().split(_GO)
                                                                      
                if len(line) == 2:
                    enc = np.asarray(line[0].strip().split(), dtype=np.float32)
                    enc = np.reshape(enc, [len(enc) // 2, 2])
                    
                    dec = np.asarray(line[1].strip().split(), dtype=np.float32)

                    # TODO: is this necessary ??
                    while len(dec) != len(enc):
                        dec = np.append(dec, _PAD)

                    enc_in.append(enc)
                    dec_out.append(dec)

            return enc_in, dec_out

def get_batch(enc, dec, batch_size, name=None):
    with tf.name_scope(name, "batch", [enc, dec, batch_size]):
        enc = tf.convert_to_tensor(enc, name="enc", dtype=tf.float32)
        dec = tf.convert_to_tensor(dec, name="dec", dtype=tf.int32)

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
if __name__ == "__main__":
    enc_in, dec_out = _load_data("./convex_hull_50_train.txt")
    print(enc_in)
