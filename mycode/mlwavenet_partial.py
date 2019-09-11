# -*- coding: utf-8 -*-
"""
This is written by https://github.com/munich-ai-labs/keras2-wavenet/blob/master/mlwavenet.py
Here's not the full version of wavenet, only the residual block. 
We want to use the gated activation function inside the residual block.

"""

from __future__ import absolute_import, division, print_function
import sys
# reload(sys)
sys.setdefaultencoding('utf-8')
import datetime
import json
import os
import re
import wave
import keras.backend as K
import numpy as np
import scipy.io.wavfile
import scipy.signal
import getopt
import codecs
from keras import layers
from keras import metrics
from keras import objectives
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.engine import Input
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

from mycode.wavenet_utils import CausalDilatedConv1D, categorical_mean_squared_error



class MLWaveNet(object):
    def __init__(self):
        # Training Configuration
        self.output_bins = 256
        self.filters = 256
        self.dilation_depth = 9
        self.stacks = 1
        self.use_bias = False
        self.res_l2 = 0
        self.final_l2 = 0
        self.use_skip_connections = True
        self.learn_all_outputs = True

        # self.sample_rate = 4410
        self.dilation_depth = 9

        # self.initial_fragment_length = 128
        # self.fragment_length = self.initial_fragment_length + self._compute_receptive_field2(self.sample_rate, self.dilation_depth, self.stacks)[0]


    # def _compute_receptive_field(self):
    #     return self._compute_receptive_field2(self.sample_rate, self.dilation_depth, self.stacks)

    # def _compute_receptive_field2(self, sample_rate, dilation_depth, stacks):
    #     receptive_field = stacks * (2 ** dilation_depth * 2) - (stacks - 1)
    #     receptive_field_ms = (receptive_field * 1000) / sample_rate
    #     return receptive_field, receptive_field_ms



    # Building the model
    def _build_model_residual_block(self, x, i, s):
        original_x = x
        # TODO: initalization, regularization?
        tanh_out = CausalDilatedConv1D(self.filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True, bias=self.use_bias, name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh', W_regularizer=l2(self.res_l2))(x)
        sigm_out = CausalDilatedConv1D(self.filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True, bias=self.use_bias, name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid', W_regularizer=l2(self.res_l2))(x)
        x = layers.Multiply()([tanh_out, sigm_out])

        res_x = layers.Conv1D(self.filters, 1, padding='same', use_bias=self.use_bias, kernel_regularizer=l2(self.res_l2))(x)
        skip_x = layers.Conv1D(self.filters, 1, padding='same', use_bias=self.use_bias, kernel_regularizer=l2(self.res_l2))(x)
        res_x = layers.Add()([original_x, res_x])
        return res_x, skip_x

    def _build_model(self,inputs,return_model=False):
        # inputs = Input(shape=(self.fragment_length, self.output_bins), name='input_part')
        out = inputs
        skip_connections = []
        out = CausalDilatedConv1D(self.filters, 2, atrous_rate=1, border_mode='valid', causal=True, name='initial_causal_conv')(out)
        for s in range(self.stacks):
            for i in range(0, self.dilation_depth + 1):
                out, skip_out = self._build_model_residual_block(out, i, s)
                skip_connections.append(skip_out)

        if self.use_skip_connections:#if not using skip, the out is the final added residual out
            out = layers.Add()(skip_connections)
        out = layers.Activation('relu')(out)
        out = layers.Conv1D(self.output_bins, 1, padding='same', kernel_regularizer=l2(self.final_l2))(out)
        out = layers.Activation('relu')(out)
        out = layers.Conv1D(self.output_bins, 1, padding='same')(out)
        if not self.learn_all_outputs:
            raise DeprecationWarning('Learning on just all outputs is wasteful, now learning only inside receptive field.')
            out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(out)  # Based on gif in deepmind blog: take last output?

        out = layers.Activation('softmax', name="output_softmax")(out)
        
        if return_model:
            model = Model(inputs, out)
            # self.receptive_field, self.receptive_field_ms = self._compute_receptive_field()
            return model
        else:
            return out

if __name__ == '__main__': 
    wavenet = MLWaveNet()
    model = wavenet._build_model()







