# -*- coding: utf-8 -*-
'''
  @Time    : 2022/11/4 9:35
  @Author  : lutingyu
  @FileName: autoEncoder.py
  @Software: PyCharm
  @Description:
'''
import keras
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np

def AEncoder(encoding_dim):
    # 自编码器输入
    input_img = Input(shape=(940,475,270))

    # 使用一个全连接网络来搭建编码器
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    # 使用一个全连接网络来对编码器进行解码
    decoded = Dense(784, activation='sigmoid')(encoded)
    # 编码器模型
    encoder = Model(input=input_img, output=encoded)
    encoder.compile(optimizer="Adam", loss='categorical_crossentropy')
    encoder.summary()
    x = np.zeros((1,940, 475, 270))
    y=encoder.predict(x)
    print(y.shape)

input_image=np.zeros((940,475,270))
AEncoder(15)
