#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import utils

class PtrNet():
    def __init__(self, is_training, config, input_):

        batch_size = input_.batch_size
        num_steps  = input_.num_steps   # for truncated backprop.
        lstm_width = config.hidden_size # size of hidden units

        enc_cell   = utils.LSTM(size=lstm_width)
        dec_cell   = utils.LSTM(size=lstm_width)

        # Encoder
        # ( fc > elu > lstm ) 
        enc_state = enc_cell.zero_state(batch_size, tf.float32)
        enc_states  = []

        with tf.variable_scope("enc"):
            for i in range(num_steps):                
                if i > 0: tf.get_variable_scope().resue_variables()

                inputs  = input_.input_data[:, i, :]

                # 2d -> lstm width 
                cell_in = utils.fc(inputs, inputs.get_shape()[-1], lstm_width, a_fn=tf.nn.elu)
                (enc_cell_out, enc_state) = enc_cell(cell_in, enc_state)

                enc_states.append(enc_state)

        # for test
        self.enc_final_state = enc_state
        
        # Decoder
        # ( fc > elu > lstm > v^t tanh(W1 e + W2 d) > softmax > argmax )
        #dec_state = enc_states[-1]

        #with tf.variable_scope("dec"):
            
            
                    
