# -*- coding: utf-8 -*-
'''
  @Time    : 2022/10/19 20:52
  @Author  : lutingyu
  @FileName: pre-processing.py
  @Software: PyCharm
  @Description: 从降维的数据中按固定比例从每类样本中随机提取一定数据的样本
'''
import numpy as np
import random
import math
import cv2


# 默认取20%
def get_samples(features, gt, per=0.6):  # 0 < per <= 1
    features = features.reshape(features.shape[0] * features.shape[1], features.shape[2])
    print(features.shape)
    gt = np.array(gt).flatten()  # 标签拍平
    print(gt.shape)
    new_features = []
    new_gt = []
    for i in range(len(np.unique(gt))):  # len(np.unique(gt))
        # print(i, len(features[gt == i]))
        # 默认取各样本总数的 per%
        print(len(list(features[gt == i])))
        num_samples = round(len(features[gt == i]) * per)
        print(num_samples)
        rnd_data = random.sample(list(features[gt == i]), num_samples)
        print(len(rnd_data))
        for k in range(len(rnd_data)):
            new_features.append(rnd_data[k])
        print(len(new_features))
        for _ in range(num_samples):
            new_gt.append(i)
        print(len(new_gt))
    new_features = np.array(new_features)
    new_gt = np.array(new_gt)
    print(new_features.shape)
    print(new_gt.shape)
    # 查看每类样本的数量，与原文对比数量,对比过了，no problem!
    print(np.unique(new_gt))
    # 打散后再截取
    data_num, _ = new_features.shape  # 得到样本数
    index = np.arange(data_num)  # 生成下标
    np.random.shuffle(index)
    new_features = new_features[index]
    new_gt = new_gt[index]
    print("打散后的数据shape", new_features.shape, new_gt.shape)

    # 开平方，截取
    edge_length = int(math.floor(math.sqrt(new_gt.shape[0])))
    print(edge_length)
    new_features = new_features[:int(math.pow(edge_length, 2)), :]
    print(new_features.shape)
    new_gt = new_gt[:int(math.pow(edge_length, 2))]
    print(new_gt.shape)
    new_features = new_features.reshape(edge_length, edge_length, 5)
    new_gt = new_gt.reshape(edge_length, edge_length, 1)
    print(new_features.shape)
    print(new_gt.shape)

    # np.save("../../datasets/WHU-Hi/subSamples/hsi_WHU-Hi_5_20per.npy",new_features)
    # np.save("../../datasets/WHU-Hi/subSamples/hsi_WHU-Hi_gt_5_20per.npy", new_gt)
    # cv2.imwrite("../../datasets/WHU-Hi/subSamples/hsi_WHU-Hi_gt_5_20per_vis.png",new_gt*10)

    for i in range(len(np.unique(new_gt))):
        print(i, np.sum(new_gt == i),np.sum(new_gt == i)/len(new_gt))
        # print(i,256.*256*np.sum(new_gt == i)/len(new_gt))
    # return new_features, new_gt


hsi = np.load("../../datasets/WHU-Hi/hsi_WHU-Hi_5.npy")
gt = np.load("../../datasets/WHU-Hi/hsi_WHU-Hi_gt_5.npy")
print(hsi.shape)
print(gt.shape)
get_samples(hsi, gt, per=0.2)
