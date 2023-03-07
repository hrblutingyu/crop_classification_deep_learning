# -*- coding: utf-8 -*-
'''
  @Time    : 2022/12/18 17:11
  @Author  : lutingyu
  @FileName: ACheng_adaboost.py
  @Software: PyCharm
  @Description:
'''
import cv2
from sklearn.model_selection import train_test_split
from sklearn import tree, datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
import pickle

from sklearn.metrics import *
from sklearn.preprocessing import scale
import scipy.io as sio
from sklearn.decomposition import PCA
import random
import datetime

# 原始图像的 列与行 or 宽与高 通道数
width = 0
height = 0
channels = 0
report = "./evaluating_indicator/mishan_report_svm.txt"

colors_dict = [(1, 1, 1), (1, 250, 2), (255, 127, 80), (22, 100, 250)]

def load_data():
    img = cv2.imread("../datasets/mishan/roi_1_clip.tif", 1)
    print(img.shape)
    gt = cv2.imread("../datasets/mishan/roi_1_label_gray.tif", 0)
    print(gt.shape)
    print(np.unique(gt))
    for i in np.unique(gt):
        print(i, len(img[gt == i]))
        print(i, len(gt[gt == i]))
    global height
    global width
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    features = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    print(features.shape)
    gt = np.array(gt).flatten()  # 标签拍平
    print(gt.shape)

    # new_features = []
    # new_features.append(features[gt==1])
    #
    # print(len(new_features))
    # print(np.array(new_features).shape)

    new_features = np.vstack(
        [features[gt == 0], features[gt == 1], features[gt == 2], features[gt == 3]])
    new_gt = np.hstack([gt[gt == 0], gt[gt == 1], gt[gt == 2], gt[gt == 3]])

    print(new_features.shape)
    print(new_gt.shape)
    return new_features, new_gt

# 默认取50%
def get_samples(features, gt, per=1.0):  # 0 < per <= 1
    new_features = []
    new_gt = []
    for i in range(len(np.unique(gt))):
        # print(i, len(features[gt == i]))
        # 默认取各样本总数的50%
        num_samples = round(len(features[gt == i]) * per)
        # print(num_samples)
        rnd_data = random.sample(list(features[gt == i]), num_samples)
        for k in range(len(rnd_data)):
            new_features.append(rnd_data[k])
        # print(len(new_features))
        for _ in range(num_samples):
            new_gt.append(i)

    return new_features, new_gt


def Classifier_svm():  # 训练并生成模型
    features, gt = load_data()

    # 存储不同降维尺度下的总体分类精度(OA)
    AAs = []
    OAs = []
    # 打开记录性能指标的文件
    Note = open(report, mode='a', encoding="utf-8")
    Note.write(str(datetime.datetime.now()) + "\t author: lutingyu\n")

    rnd_samples_features, rnd_gt = get_samples(features, gt, per=1.0)

    x_train, x_test, y_train, y_test = train_test_split(rnd_samples_features, rnd_gt, test_size=0.3, random_state=0)
    # classifier = AdaBoostClassifier(n_estimators=200, learning_rate=1.0)
    classifier = svm.SVC()
    print("training......")
    classifier.fit(x_train, y_train)  # 拟合模型
    # 混淆矩阵
    # matrix = plot_confusion_matrix(classifier, x_test, y_test,cmap=plt.cm.Blues,normalize='true')
    # plt.title('Matrix of  Random Forest')
    # plt.show()

    score = classifier.score(x_test, y_test)
    # 保存分类器
    with open("./trainedModels/svm_kernel_0.pkl", "wb") as f:
        pickle.dump(classifier, f)

    print("测试集：", score)
    X = cv2.imread("../datasets/mishan/roi_3_clip.tif", 1)
    img_h = X.shape[0]
    img_w = X.shape[1]
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    result = classifier.predict(X)
    print(result.shape)
    # 把结果转换为二维，生成图像
    result_img = result.reshape(img_h, img_w)
    cv2.imwrite("../datasets/mishan/roi_3_pred_svm.tif", result_img)


    Note.write("\t\tscore of test set : " + str(score) + "\n")
    # result = classifier.predict(features)
    #
    # # 性能度量
    test_gt = cv2.imread("../datasets/mishan/roi_3_label_new_gray.tif", 0)
    test_gt = test_gt.reshape(test_gt.shape[0] * test_gt.shape[1])

    eva_indicator, overall_accuracy, average_accuracy = evaluating_indicator(test_gt, result)
    Note.writelines(eva_indicator)
    AAs.append(average_accuracy)
    OAs.append([overall_accuracy])

    Note.write("\n------------------------------------------------------------------------------------")
    Note.write("\nAAs : " + str(AAs))
    Note.write("\nOAs : " + str(OAs))
    Note.write("\nThe lowest accuracy : " + str(np.min(OAs)) + ", channel : " + str(np.argmin(OAs) + 1))
    Note.write("\nThe highest accuracy : " + str(np.max(OAs)) + ", channel : " + str(np.argmax(OAs) + 1))
    Note.write("\n------------------------------------------------------------------------------------")
    Note.close()

def evaluating_indicator(y_true, y_pred):  # 评价指标
    classify_report = classification_report(y_true, y_pred)
    confusionMatrix = confusion_matrix(y_true, y_pred)
    overall_accuracy = accuracy_score(y_true, y_pred)
    acc_for_each_class = precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    score = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusionMatrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy(AA): {0:f}'.format(average_accuracy))
    print('overall_accuracy(OA): {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))
    print("kappa:{0:f}".format(kappa))
    eva_indicator = []  # 记录性能指标的数组
    eva_indicator.append("\t\tclassify_report :" + str(classify_report) + "\n")
    eva_indicator.append("\t\tconfusion_matrix :" + str(confusionMatrix) + "\n")
    eva_indicator.append("\t\tacc_for_each_class :" + str(acc_for_each_class) + "\n")
    eva_indicator.append("\t\t" + 'average_accuracy(AA): {0:f}'.format(average_accuracy) + "\n")
    eva_indicator.append("\t\t" + 'overall_accuracy(OA): {0:f}'.format(overall_accuracy) + "\n")
    eva_indicator.append('\t\tscore: {0:f}'.format(score) + "\n")
    eva_indicator.append("\t\tkappa:{0:f}".format(kappa) + "\n")
    return eva_indicator, overall_accuracy, average_accuracy

Classifier_svm()