import tensorflow as tf
import argparse
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,\
    Reshape,Permute,Activation,Flatten,Dropout,Dense,ZeroPadding2D,Input,AveragePooling2D,Add,Conv2DTranspose,Lambda
from keras.layers.merge import concatenate, add, maximum, subtract, multiply, dot

from FCSNet.metrics import *

# 以下为ResNet实现
# ResNet原文中的直联identity_block
def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# ResNet原文中的残差bottleneck_block
def convolution_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c',
               kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer='glorot_uniform')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# 论文中的第一个网络,backbone为ResNet50-------------------------
# 分类结果4个类别，3种作物 + 背景-------------------------------
# input_shape:如果是rgb数据则为(256, 256, 3),如果是添加了NDVI,则为(256, 256, 4)
def FCSNetResNet50(input_shape=(256, 256, 3), classes=4):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    # custom，添加了ZeroPadding2D，保证下一层的输入为16*16，否则为15*15
    X = ZeroPadding2D((1, 1))(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolution_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolution_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    stage_2_output=X

    X = convolution_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    stage_3_output=X

    X = convolution_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    stage_4_output = X
    # 8,8,2048
    # 原网络ResNet50中的最后一层全局池化层,在FCSNet中被取消了。
    # X = AveragePooling2D((2, 2), name='avg_pool')(X)
    # 自定义 在FCSNet中 将全局池化替换为conv2d 维度不变
    X=Conv2D(2048, (3, 3), padding='same', kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X=Conv2D(2048, (3, 3), padding='same', kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Classifying layers.
    X = Conv2D(classes, (1, 1), strides=(1, 1), activation='linear')(X)
    X = BatchNormalization()(X)

    stage_3_output=Conv2D(classes, (1, 1), strides=(1, 1), activation='linear',name='stage_3_output')(stage_3_output)
    stage_3_output = BatchNormalization()(stage_3_output)
    X = Conv2DTranspose(classes, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal")(X)
    X = add([X, stage_3_output])
    # X = Activation('relu')(X)
    # 16,16,4

    stage_2_output = Conv2D(classes, (1, 1), strides=(1, 1), activation='linear', name='stage_2_output')(stage_2_output)
    X = Conv2DTranspose(classes, (3, 3), strides=(2, 2), padding='same', kernel_initializer="he_normal")(X)
    X = add([X, stage_2_output])
    # X = Activation('relu')(X)
    # 32,32,4

    # 8倍上采样,分别尝试反卷积，upsampling, image.resize
    # X = Conv2DTranspose(classes, (3, 3), strides=(8, 8), padding='same', kernel_initializer="he_normal")(X)
    # X = Activation('softmax')(X)
    # upsampling
    X = UpSampling2D(size=(8, 8), interpolation="bilinear",name="upsamping_1")(X)
    X = Activation('softmax')(X)
    # image.resize
    # X = Lambda(lambda X: tf.image.resize(X, (X.shape[1] * 8, X.shape[2] * 8)))(X)
    # X = Activation('softmax')(X)

    # X = Flatten()(X)
    # X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer='glorot_uniform')(X)

    # # --- 自定义 第一次反卷积 输入为4*4,经过第一次反卷积把分辨率上采样到8*8，并且与stage_4_output的输出(8*8)执行加操作
    # X = Conv2DTranspose(2048, (3,3), strides=(2, 2), padding='same', activation='relu')(X)
    # X=add([X,stage_4_output])
    # #______至此，输出为8*8@2048_____________________________________________________________________________________
    #
    # # ----- 自定义 第二次反卷积 输入8*8,经过第二次反卷积把分辨率上采样到16*16，并且与stage_3_output的输出(16*16)执行加操作
    # X = Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
    # X = add([X, stage_3_output])
    # #-------至此，输出为16*16@1024______________________________________________________________________
    #
    # # ----- 自定义 第三次反卷积 输入16*16,经过第三次反卷积把分辨率上采样到32*32，并且与stage_2_output的输出(32*32)执行加操作
    # X = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(X)
    # X = add([X, stage_2_output])
    # #-------至此，输出为32*32@512______________________________________________________________________
    #
    # # ----- 自定义 第四次反卷积 输入32*32,经过第四次反卷积把分辨率上采样到256*256，恢复原图，不再执行加操作
    # # X = Conv2DTranspose(classes, (3, 3), strides=(8, 8), padding='same', activation='softmax')(X)
    # #-------至此，输出为256*256@512______________________________________________________________________
    # # 是否使用如下的conv?跑完基础的模型后再定夺
    # # 使用8倍上采样代替第四次卷积,再添加一个卷积层分类
    # X = UpSampling2D(size=(8, 8), interpolation="bilinear",
    #                    name="upsamping_1")(X)
    # X = Conv2D(classes, (3, 3), activation="softmax",padding="same", name="conv_softmax")(X)

    model = Model(inputs=X_input, outputs=X, name='FCSNet_ResNet50')
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy', mean_iou, dice_coef])
    model.summary(line_length=150)
    return model
FCSNetResNet50()