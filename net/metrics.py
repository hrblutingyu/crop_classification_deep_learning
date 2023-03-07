"""
The implementation of some metrics based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.metrics import confusion_matrix
import cv2

def mean_iou(y_true, y_pred):
    num_classes = y_pred.get_shape().as_list()[-1]

    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    ious = list()
    for i in range(num_classes):
        yti = tf.equal(y_true, i)
        ypi = tf.equal(y_pred, i)

        intersection = tf.reduce_sum(tf.cast(tf.logical_and(yti, ypi), tf.float32), axis=[1, 2])
        union = tf.reduce_sum(tf.cast(tf.logical_or(yti, ypi), tf.float32), axis=[1, 2])

        iou = tf.where(tf.equal(union, 0.), tf.ones_like(union), intersection / union)

        ious.append(iou)
    ious = tf.stack(ious, axis=-1)
    # return tf.reduce_sum(ious, axis=-1) / num_classes
    return tf.reduce_sum(ious, axis=-1) / num_classes

# 对于分割过程中的评价标准主要采用Dice相似系数(Dice Similariy Coefficient,DSC),
# Dice系数是一种集合相似度度量指标,通常用于计算两个样本的相似度,值的范围 [公式] ,
# 分割结果最好时值为 1 ,最差时值为 0
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# -*- coding: utf-8 -*-
# author： admin
# datetime： 2021/12/16 17:19
# ide： PyCharm
# https://github.com/bubbliiiing/unet-keras/blob/9adaa621502699693c1c3c01f4e596f1a9e6930e/utils/utils_metrics.py

# 将测试集的预测结果保存到本地，再与标签进行mIoU计算
import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from PIL import Image


def Iou_score(smooth=1e-5, threhold=0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())
        intersection = backend.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        union = backend.sum(y_true[..., :-1] + y_pred, axis=[0, 1, 2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score

    return _Iou_score


def f_score(beta=1, smooth=1e-5, threhold=0.5):
    def _f_score(y_true, y_pred):
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())

        tp = backend.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        fp = backend.sum(y_pred, axis=[0, 1, 2]) - tp
        fn = backend.sum(y_true[..., :-1], axis=[0, 1, 2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score

    return _f_score


# 设标签宽W，长H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    # gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    # pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]
    #
    gt_imgs = [join(gt_dir, x ) for x in png_name_list]
    pred_imgs = [join(pred_dir, x ) for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )
    # ------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
              + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision[ind_class] * 100, 2)))

    # -----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))