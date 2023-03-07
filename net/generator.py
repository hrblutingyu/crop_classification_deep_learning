# -*- coding: utf-8 -*-
'''
  @Time    : 2022/9/30 7:53
  @Author  : lutingyu
  @FileName: generator.py
  @Software: PyCharm
  @Description: 数据生成器，喂饼机，吊炉饼
'''
import numpy as np
import random
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, scale

# 0为背景,定义类别，one hot编码----------------------------------------------
img_w = 256
img_h = 256
# n_label = 22+1
n_label = 23
classes = []
for i in range(n_label):
    classes.append(i)
print(classes)
labelencoder = LabelEncoder()
labelencoder.fit(classes)
# ---------------------------------------------------------------------------

filepath = '../../datasets/WHU-Hi/train/'


def get_train_test(data_path):
    train_set_num = len(os.listdir(data_path + 'train/hsi'))
    val_set_num = len(os.listdir(data_path + 'val/hsi'))
    test_set_num = len(os.listdir(data_path + 'test/hsi'))
    return train_set_num, val_set_num, test_set_num


def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)
    # 测试集不做增强
    elif mode == 'test':
        x_data_gen_args = dict()
        y_data_gen_args = dict()

    else:
        print("mode arg 'train' or 'val'.")
        return -1

    return x_data_gen_args, y_data_gen_args


# mode:train/val/test
# 三种数据集合组织形式为data下有train/val/test文件夹,三个文件夹下分别有images,labels文件夹
def data_generator(data_path, batch_size, mode):
    x_path = os.path.join(data_path, mode + "/hsi")
    y_path = os.path.join(data_path, mode + "/hsi_gt")

    x_files = os.listdir(x_path)
    y_files = os.listdir(y_path)
    print(len(x_files))
    print(len(y_files))

    #
    # mode: train or val or test
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)
    # print(x_data_gen_args)
    # print(y_data_gen_args)
    data_size = len(x_files)
    print(data_size)
    #
    x = []
    y = []
    #
    while True:
        random.shuffle(x_files)
        for i in range(data_size):
            img_name = x_files[i]
            img_name = os.path.join(x_path, img_name)
            # print(img_name)

            x_img = np.load(img_name)
            # print(x_img.shape)
            # 是否作归一化处理，不作就注释掉------------------------------------
            # img_width=x_img.shape[0]
            # img_height = x_img.shape[1]
            # channels = x_img.shape[2]
            # # print(img_height,img_width,channels)
            # x_img=x_img.reshape(img_height*img_width,channels)
            # x_img = scale(x_img)
            # x_img = x_img.reshape(img_height,img_width,channels)
            # print(x_img.shape)
            # 归一化结束------------------------------------------

            # # 标签在labels文件夹下，与在图像文件同名,但在不同文件夹下,只需要更改到标签路径下，即可找到对应的标签
            lbl_name = x_files[i]
            lbl_name = os.path.join(y_path, lbl_name)
            # print(lbl_name)

            # y_img = cv2.imread(lbl_name, cv2.IMREAD_GRAYSCALE)
            y_img = np.load(lbl_name)
            # print(y_img.shape)
            # # y_img = y_img.reshape((img_w * img_h,))

            x.append(x_img)
            y.append(y_img)

            if len(x) == batch_size:
                x = np.array(x)
                y = np.array(y)
                # print(x.shape,y.shape)
                y = y.flatten()  # 拍平
                y = labelencoder.transform(y)
                y = to_categorical(y, num_classes=n_label)  # 编码输出便签
                y = y.reshape((batch_size, img_w, img_h, n_label))
                # 如果想存储增强后的标签数据，注释掉上4行代码，加入下面的一行
                _ = np.zeros(batch_size)
                seed = random.randrange(1, 1000)

                x_gen = x_data_gen.flow(np.array(x), _, batch_size=batch_size, seed=seed)
                y_gen = y_data_gen.flow(np.array(y), _, batch_size=batch_size, seed=seed)

                # 迭代
                x_result, _ = next(x_gen)
                y_result, _ = next(y_gen)
                yield x_result, y_result
                x = []
                y = []


def test_data_generator(test_x_path="../../datasets/WHU-Hi/test/hsi/"):
    x_files = os.listdir(test_x_path)
    for file in x_files:
        # img = cv2.imread(test_x_path + x_files[i], 1)
        # img = img / 255.
        # img = np.reshape(img, (1,) + img.shape)
        # print(img.shape)
        # print(test_x_path + x_files[i])
        img = np.load(test_x_path + file)
        print(file)
        img = np.reshape(img, (1,) + img.shape)  # (w,h,c) to (batchsize,w,h,c) 否则报维度错误
        # print(img.shape)
        yield img

# data_generator(data_path="../../datasets/WHU-Hi/", batch_size=16, mode="train")
# test_data_generator()
