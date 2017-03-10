import tensorflow as tf

from utils import conv, fc, flatten



class QFunction:
    def __init__(self, input_shape, action_n, scope):
        self.scope = scope
        self.batch_size = input_shape[0]
        self.depth = input_shape[-1]
        self.a = action_n
        self.in_shape = input_shape

    def _model(self, X, reuse=False, s_bias=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            
            conv1 = conv(X, [4, 4, self.depth, 32], [1, 4, 4, 1],
                         activation_fn=tf.nn.relu, scope="conv1")
            conv2 = conv(conv1, [4, 4, 32, 64], [1, 2, 2, 1],
                         activation_fn=tf.nn.relu, scope="conv2")
            conv3 = conv(conv2, [3, 3, 64, 64], [1, 1, 1, 1],
                         activation_fn=tf.nn.relu, scope="conv3")

            flt, dim = flatten(conv3)

            fc1 = fc(flt, dim, 512, scope="fc1")
                        
            with tf.variable_scope("fc2"):
                W = tf.get_variable("W", [512, self.a], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=0.02))
            
        # shared bias
        with tf.variable_scope("shared", reuse=s_bias):
            b = tf.get_variable("shared_b", [self.a], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
            
            return tf.nn.bias_add(tf.matmul(fc1, W), b)
            

            # for cartpole-v0 test
            #fc1 = fc(X, self.in_shape[1], 100, activation_fn=tf.nn.relu, scope="fc1")
            #fc2 = fc(fc1, 100, 32, activation_fn=tf.nn.relu, scope="fc2")
            #return fc(fc2, 32, self.a, scope="fc3")
                     
    def __call__(self, X, reuse=False, s_bias=True):
        
        return self._model(X, reuse=reuse, s_bias=s_bias)
