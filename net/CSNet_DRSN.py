# -*- coding: utf-8 -*-
'''
  @Time    : 2022/6/2 15:17
  @Author  : lutingyu
  @FileName: FCSNet_DRSN.py.py
  @Software: PyCharm
  deep residual shrinkage netowrks
'''
from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
K.set_learning_phase(1)

def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs,1),1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs,-1)
    inputs = K.spatial_3d_padding(inputs, ((0,0),(0,0),(pad_dim,pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)

# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    for i in range(nb_blocks):
        identity = residual

        if not downsample:
            downsample_strides = 1

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides),
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)

        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)

        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])

        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1, 1), strides=(2, 2))(identity)

        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels': in_channels, 'out_channels': out_channels})(
                identity)

        residual = keras.layers.add([residual, identity])

    return residual
# define and train a model
input_shape=(256,256,3)
inputs = Input(shape=input_shape)
net = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
net = residual_shrinkage_block(net, 1, 8, downsample=True)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = GlobalAveragePooling2D()(net)
outputs = Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
model = Model(inputs=inputs, outputs=outputs)
model.summary()