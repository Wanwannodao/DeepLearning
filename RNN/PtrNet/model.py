#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import utils

class PtrNet():
    def __init__(self, is_training, config, input_):

        batch_size = input_.batch_size
        num_steps  = input_.num_steps   # for truncated backprop.
        lstm_width = config.hidden_size # size of hidden units
        in_shape   = input_.input_data.get_shape()

        enc_cell   = utils.LSTM(size=lstm_width)
        dec_cell   = utils.LSTM(size=lstm_width)

        # Encoder
        # ( fc > elu > lstm ) 
        enc_state = enc_cell.zero_state(batch_size, tf.float32)
        enc_states  = []

        with tf.variable_scope("enc"):
            for i in range(num_steps):                
                if i > 0: tf.get_variable_scope().reuse_variables()

                enc_inputs  = input_.input_data[:, i, :]

                # 2d -> lstm width 
                enc_cell_in = utils.fc(enc_inputs, enc_inputs.get_shape()[-1], lstm_width, a_fn=tf.nn.elu)
                (enc_cell_out, enc_state) = enc_cell(enc_cell_in, enc_state)

                enc_states.append(enc_state)

        # for test
        # self.enc_final_state = enc_state
        
        # Decoder
        # ( fc > elu > lstm > v^t tanh(W1 e + W2 d) > softmax > argmax )
        print(in_shape)
        dec_state  = enc_states[-1]
        dec_inputs = tf.constant(-2.0,
                                 shape=[batch_size, 2],
                                 dtype=tf.float32) # start symbol
        
        self.C = []
        with tf.variable_scope("dec"):
            for i in range(num_steps):
                if i > 0: tf.get_variable_scope().reuse_variables()

                dec_cell_in = utils.fc(dec_inputs, dec_inputs.get_shape()[-1], lstm_width, a_fn=tf.nn.elu)

                (dec_cell_out, dec_state) = dec_cell(dec_cell_in, dec_state)

                # W1, W2 are square matrixes (SxS)
                # where S is the size of hidden states
                W1 = tf.get_variable("W1", [lstm_width, lstm_width], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.02))
                W2 = tf.get_variable("W2", [lstm_width, lstm_width], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.02))
                # v is a vector (S)
                v  = tf.get_variable("v", [lstm_width], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))

                # W2 (SxS) d_i (S) = W2d (S)
                W2d = tf.matmul(dec_state.h, W2)

                # u_i (n)
                u_i = []
                
                for j in range(num_steps):
                    # W1 (SxS) e_j (S) = W1e (S)
                    # t = tanh(W1e + W2d) (S)
                    t    = tf.tanh( tf.matmul(enc_states[j].h, W1) + W2d )
                    # v^T (S) t (S) = U_ij (1)  
                    u_ij = v*t

                    u_i.append(u_ij)

                u_i   = tf.stack(u_i) # asarray
 
                probs = tf.nn.softmax(u_i)
                C_i   = tf.argmax(probs)

                #dec_inputs = inputs_.input_data[:, C_i, :]
                
                self.C.append(C_i)
                
                
                    
                                     
            
                    
