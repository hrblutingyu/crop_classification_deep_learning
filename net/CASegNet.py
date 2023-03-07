# -*- coding: utf-8 -*-
'''
  @Time    : 2022/10/2 20:06
  @Author  : lutingyu
  @FileName: CA-SegNet.py
  @Software: PyCharm
  @Description: channel attention segnet
'''
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
# from layers import *
from metrics import *

img_w = 256
img_h = 256
channels = 5
# 0为背景,定义类别，one hot编码
# n_label = 22+1
n_label = 23
classes = []
for i in range(n_label):
    classes.append(i)
print(classes)
labelencoder = LabelEncoder()
labelencoder.fit(classes)

# 通道注意力机制
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)  # Reshape: width,height,depth
    # assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool, max_pool])  # 处理后的结果相加
    cbam_feature = Activation('sigmoid')(cbam_feature)  # 获得各通道的权重图

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])
def get_CASegNet(img_h, img_w, channels):
    # encoder
    inputs = Input(shape=(img_h, img_w, channels))

    conv_1 = Convolution2D(64, (3, 3), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (3, 3), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)
    # 加入1 个channel attention
    ca_1_1 = channel_attention(conv_2)

    # -------------------------------
    pool_1 = MaxPooling2D(2)(ca_1_1)

    conv_3 = Convolution2D(128, (3, 3), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (3, 3), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)
    # 加入1个channel attention
    ca_2_1 = channel_attention(conv_4)
    # -------------------------------
    pool_2 = MaxPooling2D(2)(ca_2_1)

    conv_5 = Convolution2D(256, (3, 3), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (3, 3), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (3, 3), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    # 加入1个channel attention
    ca_3_1 = channel_attention(conv_7)
    # -------------------------------
    pool_3 = MaxPooling2D(2)(ca_3_1)

    conv_8 = Convolution2D(512, (3, 3), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (3, 3), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (3, 3), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    # 加入1个channel attention
    ca_4_1 = channel_attention(conv_10)
    # -------------------------------
    pool_4 = MaxPooling2D(2)(ca_4_1)

    conv_11 = Convolution2D(512, (3, 3), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (3, 3), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (3, 3), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)
    # 加入1 个channel attention
    ca_5_1 = channel_attention(conv_13)
    # -------------------------------
    pool_5 = MaxPooling2D(2)(ca_5_1)

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
    # 跳层1 (16,16,1024)
    merge_1 = concatenate([ca_5_1, conv_16], axis=3)
    unpool_2 = UpSampling2D(2)(merge_1)

    conv_17 = Convolution2D(512, (3, 3), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (3, 3), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (3, 3), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    # 加入channel attention
    # ca_7 = channel_attention(conv_19)
    # -------------------------------
    # 跳层2 (32,32,768)
    merge_2 = concatenate([ca_4_1, conv_19], axis=3)

    unpool_3 = UpSampling2D(2)(merge_2)

    conv_20 = Convolution2D(256, (3, 3), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (3, 3), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (3, 3), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    # -----------------------------------------------------
    # 跳层3 (64, 64, 384)
    merge_3 = concatenate([ca_3_1, conv_22], axis=3)
    unpool_4 = UpSampling2D(2)(merge_3)

    conv_23 = Convolution2D(128, (3, 3), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (3, 3), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    # 加入channel attention
    # ca_9 = channel_attention(conv_24)
    # -------------------------------
    # 跳层4 (128,128,192)
    merge_4 = concatenate([ca_2_1, conv_24], axis=3)

    unpool_5 = UpSampling2D(2)(merge_4)
    # 跳层 5 (256,256,256)
    merge_5 = concatenate([ca_1_1, unpool_5], axis=3)

    conv_25 = Convolution2D(64, (3, 3), padding="same")(merge_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_label, (1, 1), padding="valid")(conv_25)
    outputs = Activation("softmax")(conv_26)


    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', mean_iou, dice_coef])
    model.summary(line_length=150)
    return  model
get_CASegNet(img_h=256, img_w=256, channels=5)
