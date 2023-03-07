# -*- coding: utf-8 -*-
'''
  @Time    : 2022/10/9 8:17
  @Author  : lutingyu
  @FileName: SegNet.py
  @Software: PyCharm
  @Description:
'''
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
# from layers import *
from metrics import *

img_w = 256
img_h = 256
# channels = 5
# 0为背景,定义类别，one hot编码
# n_label = 22+1
n_label = 23
classes = []
for i in range(n_label):
    classes.append(i)
print(classes)
labelencoder = LabelEncoder()
labelencoder.fit(classes)

def get_SegNet(img_h, img_w, channels):
    # encoder
    inputs = Input(shape=(img_h, img_w, channels))

    conv_1 = Convolution2D(64, (3, 3), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (3, 3), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1 = MaxPooling2D(2)(conv_2)

    conv_3 = Convolution2D(128, (3, 3), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (3, 3), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2 = MaxPooling2D(2)(conv_4)

    conv_5 = Convolution2D(256, (3, 3), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (3, 3), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (3, 3), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3 = MaxPooling2D(2)(conv_7)

    conv_8 = Convolution2D(512, (3, 3), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (3, 3), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (3, 3), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4 = MaxPooling2D(2)(conv_10)

    conv_11 = Convolution2D(512, (3, 3), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (3, 3), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (3, 3), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)


    pool_5 = MaxPooling2D(2)(conv_13)

    # decoder
    unpool_1 = UpSampling2D(2)(pool_5)

    conv_14 = Convolution2D(512, (3, 3), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (3, 3), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (3, 3), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = UpSampling2D(2)(conv_16)

    conv_17 = Convolution2D(512, (3, 3), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (3, 3), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (3, 3), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)


    unpool_3 = UpSampling2D(2)(conv_19)

    conv_20 = Convolution2D(256, (3, 3), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (3, 3), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (3, 3), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = UpSampling2D(2)(conv_22)

    conv_23 = Convolution2D(128, (3, 3), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (3, 3), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = UpSampling2D(2)(conv_24)

    conv_25 = Convolution2D(64, (3, 3), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_label, (1, 1), padding="valid")(conv_25)
    outputs = Activation("softmax")(conv_26)

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', mean_iou, dice_coef])
    model.summary(line_length=150)
    return  model
# get_SegNet(256,256,5)