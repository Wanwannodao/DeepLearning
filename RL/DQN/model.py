import tensorflow as tf

from utils import conv, fc, flatten

class QFunction:
    def __init__(self, input_shape, action_n, scope):
        self.scope = scope
        self.batch_size = input_shape[0]
        self.depth = input_shape[-1]
        self.a = action_n

    def _model(self, X, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            conv1 = conv(X, [4, 4, self.depth, 32], [1, 4, 4, 1],
                         activation_fn=tf.nn.relu, scope="conv1")
            conv2 = conv(conv1, [4, 4, 32, 64], [1, 2, 2, 1],
                         activation_fn=tf.nn.relu, scope="conv2")
            conv3 = conv(conv2, [3, 3, 64, 64], [1, 1, 1, 1],
                         activation_fn=tf.nn.relu, scope="conv3")

            flt, dim = flatten(conv3)

            fc1 = fc(flt, dim, 512, scope="fc1")

            return fc(fc1, 512, self.a, scope="fc2")
    def __call__(self, X, reuse=False):
        
        return self._model(X, reuse=reuse)
