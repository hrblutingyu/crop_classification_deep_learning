# -*- coding: utf-8 -*-
# author： admin
# datetime： 2021/12/12 16:41 
# ide： PyCharm
# https://blog.csdn.net/yx123919804/article/details/104811087

from keras.layers import *
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras import backend as K
import cv2
import tensorflow as tf

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def fcn_8s(num_classes, input_shape):

    img_input = Input(shape=input_shape, name="input")

    conv_1 = Conv2D(32, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_1")(img_input)
    max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_1")(conv_1)

    conv_2 = Conv2D(64, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_2")(max_pool_1)
    max_pool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_2")(conv_2)

    conv_3 = Conv2D(128, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_3")(max_pool_2)
    max_pool_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_3")(conv_3)

    conv_4 = Conv2D(256, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_4")(max_pool_3)
    max_pool_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_4")(conv_4)

    conv_5 = Conv2D(512, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_5")(max_pool_4)
    max_pool_5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_5")(conv_5)

    # max_pool_5 转置卷积上采样 2 倍和 max_pool_4 一样大
    up6 = Conv2DTranspose(256, kernel_size=(3, 3),
                                       strides=(2, 2),
                                       padding="same",
                                       kernel_initializer="he_normal",
                                       name="upsamping_6")(max_pool_5)

    _16s = add([max_pool_4, up6])

    # _16s 转置卷积上采样 2 倍和 max_pool_3 一样大
    up_16s = Conv2DTranspose(128, kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding="same",
                                          kernel_initializer="he_normal",
                                          name="Conv2DTranspose_16s")(_16s)

    _8s = add([max_pool_3, up_16s])

    # _8s 上采样 8 倍后与输入尺寸相同
    up7 = UpSampling2D(size=(8, 8), interpolation="bilinear",
                                    name="upsamping_7")(_8s)

    # 这里 kernel 也是 3 * 3, 也可以同 FCN-32s 那样修改的
    conv_7 = Conv2D(num_classes, kernel_size=(3, 3), activation="softmax",
                                 padding="same", name="conv_7")(up7)

    model = Model(img_input, conv_7)

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])
    return model
def fcn_8s_vgg16(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):

    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_3_out = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(block_3_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_4_out = MaxPooling2D()(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(block_4_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, x)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    # Convolutinalized fully connected layer.
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classifying layers.
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)

    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_3_out)
    block_3_out = BatchNormalization()(block_3_out)

    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_4_out)
    block_4_out = BatchNormalization()(block_4_out)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 8, x.shape[2] * 8)))(x)

    x = Activation('softmax')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])

    return model
def fcn_16s(num_classes, input_shape):

    img_input = Input(shape=input_shape, name="input")

    conv_1 = Conv2D(32, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_1")(img_input)
    max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_1")(conv_1)

    conv_2 = Conv2D(64, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_2")(max_pool_1)
    max_pool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_2")(conv_2)

    conv_3 = Conv2D(128, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_3")(max_pool_2)
    max_pool_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_3")(conv_3)

    conv_4 = Conv2D(256, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_4")(max_pool_3)
    max_pool_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_4")(conv_4)

    conv_5 = Conv2D(512, kernel_size=(3, 3), activation="relu",
                                 padding="same", name="conv_5")(max_pool_4)
    max_pool_5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                                        name="max_pool_5")(conv_5)
    # max_pool_5 转置卷积上采样 2 倍至 max_pool_4 一样大
    up6 = Conv2DTranspose(256, kernel_size=(3, 3),
                                       strides=(2, 2),
                                       padding="same",
                                       kernel_initializer="he_normal",
                                       name="upsamping_6")(max_pool_5)

    _16s = add([max_pool_4, up6])

    # _16s 上采样 16 倍后与输入尺寸相同
    up7 = UpSampling2D(size=(16, 16), interpolation="bilinear",
                                    name="upsamping_7")(_16s)

    # 这里 kernel 也是 3 * 3, 也可以同 FCN-32s 那样修改的
    conv_7 = Conv2D(num_classes, kernel_size=(3, 3), activation="softmax",
                                 padding="same", name="conv_7")(up7)
    model = Model(img_input, conv_7)

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])
    return model
def fcn_32s(num_classes, input_shape):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
                                  padding="same", input_shape=input_shape,
                                  name="conv_1"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1"))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu",
                                  padding="same", name="conv_2"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2"))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu",
                                  padding="same", name="conv_3"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_3"))

    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu",
                                  padding="same", name="conv_4"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_4"))

    model.add(Conv2D(512, kernel_size=(3, 3), activation="relu",
                                  padding="same", name="conv_5"))
    # 第 5 组 MaxPool2D
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5"))
    model.add(UpSampling2D(size=(32, 32), interpolation="nearest", name="upsamping_6"))

    # 这里只有一个卷积核, 可以把 kernel_size 改成 1 * 1, 也可以是其他的, 只是要注意 padding 的尺寸
    # 也可以放到 upsamping_6 的前面, 试着改一下尺寸和顺序看一下效果
    # 这里只是说明问题, 尺寸和顺序不一定是最好的
    model.add(Conv2D(num_classes, kernel_size=(3, 3), activation="softmax",
                                  padding="same", name="conv_7"))

    model.compile(optimizer="adam",loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])
    return model

