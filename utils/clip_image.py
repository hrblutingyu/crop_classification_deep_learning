# -*- coding: utf-8 -*-
# author： admin
# datetime： 2021/12/22 8:53 
# ide： PyCharm
# 将原始图像按一定尺寸切图
import numpy as np
import math
import os
from tqdm import *
from sklearn.metrics import *

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2


# 默认切图大小为256*256
img_cell_w = 256
img_cell_h = 256
img_cell_path = "../datasets/mishan/pred/images/"
# 预测的灰度图像存储路径
img_cell_pred_path = "../datasets/mishan/pred/pred/"
# 预测的彩色图像存储路径
img_cell_pred_path_rgb = "../datasets/mishan/pred/pred_rgb/"


# 标准切图时x/y上最后一列/行图像与上一张的重叠量,需要提前计算好
# def get_overlab(src, x_step=256, y_step=256):
#     size = cv2.imread(src).shape
#     img_h, img_w = size[0], size[1]
#     x_overlap = math.floor(img_w / 256) * 256 - (img_w - x_step)
#     y_overlap = math.floor(img_h / 256) * 256 - (img_h - y_step)
#     return x_overlap, y_overlap

# 128 为重叠1/2区域裁剪
def get_overlab(src, x_step=128, y_step=128):
    img = cv2.imread(src)
    size = img.shape
    img_h, img_w = size[0], size[1]

    x_overlap = (math.floor(img_w / x_step) - 1) * x_step - (img_w - img_cell_w) + img_cell_w / 4
    y_overlap = (math.floor(img_h / y_step) - 1) * y_step - (img_h - img_cell_h) + img_cell_h / 4

    print(x_overlap, y_overlap)
    return x_overlap, y_overlap


def get_image_info(img_path):
    if img_path == "":
        print("error,img_path is empty")
        return -1

    img = cv2.imread(img_path, 1)
    size = img.shape
    img_h, img_w = size[0], size[1]

    return img_h, img_w, img


def get_img_cell_quarter(img_cell_w, img_cell_h, src_img_path):  # 1/4重叠裁剪
    img_h, img_w, img = get_image_info(src_img_path)
    # 步长
    x_step = int((img_cell_w - img_cell_w / 4))
    y_step = int((img_cell_h - img_cell_h / 4))

    cell_x_num = img_w / x_step
    cell_y_num = img_h / y_step

    cell_x_num = math.ceil(cell_x_num)
    cell_y_num = math.ceil(cell_y_num)
    # 向上取整
    # print(img_w, cell_x_num)
    # print(img_h, cell_y_num)
    for k in range(cell_y_num):
        for i in range(cell_x_num):
            # print(i)
            img_cell = np.zeros(shape=(img_cell_w, img_cell_h, 3))
            # print(i * x_step + img_cell_w)
            # print(k * y_step + img_cell_h)
            if k != (cell_y_num - 1):
                if i != (cell_x_num - 1):
                    img_cell = img[k * y_step:k * y_step + img_cell_h, i * x_step:i * x_step + img_cell_w, :]
                else:
                    img_cell = img[k * y_step:k * y_step + img_cell_h, img_w - img_cell_w - 1:img_w - 1, :]
            else:
                if i != (cell_x_num - 1):
                    img_cell = img[img_h - img_cell_h - 1:img_h - 1, i * x_step:i * x_step + img_cell_w, :]
                else:
                    img_cell = img[img_h - img_cell_h - 1:img_h - 1, img_w - img_cell_w - 1:img_w - 1, :]

            cv2.imwrite(img_cell_path + "%d" % k + "_%d" % i + ".png", img_cell)


# 标准裁剪256*256 没有重叠
def get_img_cell(img_cell_w, img_cell_h, src_img_path):
    img_h, img_w, img = get_image_info(src_img_path)
    # 没有重叠区域
    x_step = int(img_cell_w)
    y_step = int(img_cell_h)

    cell_x_num = img_w / x_step
    cell_y_num = img_h / y_step

    cell_x_num = math.ceil(cell_x_num)
    cell_y_num = math.ceil(cell_y_num)
    for k in range(cell_y_num):
        for i in range(cell_x_num):
            # print(i)
            img_cell = np.zeros(shape=(img_cell_w, img_cell_h, 3))
            # print(i * x_step + img_cell_w)
            # print(k * y_step + img_cell_h)
            if k != (cell_y_num - 1):
                if i != (cell_x_num - 1):
                    img_cell = img[k * y_step:k * y_step + img_cell_h, i * x_step:i * x_step + img_cell_w, :]
                else:
                    img_cell = img[k * y_step:k * y_step + img_cell_h, img_w - img_cell_w - 1:img_w - 1, :]
            else:
                if i != (cell_x_num - 1):
                    img_cell = img[img_h - img_cell_h - 1:img_h - 1, i * x_step:i * x_step + img_cell_w, :]
                else:
                    img_cell = img[img_h - img_cell_h - 1:img_h - 1, img_w - img_cell_w - 1:img_w - 1, :]

            cv2.imwrite(img_cell_path + "%d" % k + "_%d" % i + ".png", img_cell)


# 1/2重叠裁剪
def get_img_cell_half(img_cell_w, img_cell_h, src_img_path):
    img_h, img_w, img = get_image_info(src_img_path)
    # 步长
    x_step = int((img_cell_w - img_cell_w / 2))
    y_step = int((img_cell_h - img_cell_h / 2))

    cell_x_num = img_w / x_step
    cell_y_num = img_h / y_step

    cell_x_num = math.floor(cell_x_num)
    cell_y_num = math.floor(cell_y_num)
    # 向上取整
    # print(img_w, cell_x_num)
    # print(img_h, cell_y_num)
    print("clipping images")
    for k in tqdm(range(cell_y_num)):  # cell_y_num
        for i in range(cell_x_num):
            if k != (cell_y_num - 1):
                if i != (cell_x_num - 1):  # cell_x_num - 1
                    img_cell = img[k * y_step:k * y_step + img_cell_h, i * x_step:i * x_step + img_cell_w, :]
                else:
                    img_cell = img[k * y_step:k * y_step + img_cell_h, img_w - img_cell_w:img_w + 1, :]
            else:
                if i != (cell_x_num - 1):
                    img_cell = img[img_h - img_cell_h:img_h + 1, i * x_step:i * x_step + img_cell_w, :]
                else:
                    img_cell = img[img_h - img_cell_h:img_h + 1, img_w - img_cell_w:img_w + 1, :]
            cv2.imwrite(img_cell_path + "%d" % k + "_%d" % i + ".tif", img_cell)


# 按1/2区域重叠拼接
def mosaic_img_cell_2(img_cell_mosaic, merge_img_path):
    file_list = os.listdir(img_cell_mosaic)
    col_ids = []
    row_ids = []
    for file in file_list:
        file_name = os.path.splitext(file)[0]
        col_id = int(file_name.split("_")[0])
        row_id = int(file_name.split("_")[1])
        col_ids.append(col_id)
        row_ids.append(row_id)
    col_imgs = []
    x_overlap, y_overlap = get_overlab(src_path)

    print("merging images")
    for k in tqdm(range(len(np.unique(col_ids)))):  # len(np.unique(col_ids)))
        row_imgs = []
        for i in range(len(np.unique(row_ids))):  # len(np.unique(row_ids))
            if k == 0:
                if i == 0:  # 每行的第一张图片取3/4大小
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[:int(img_cell_h * 3 / 4), :int(img_cell_w * 3 / 4), :]
                elif i != np.max(row_ids):  # 不是第一张也不是最后一张取1/4到3/4
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[:int(img_cell_h * 3 / 4), int(img_cell_w / 4):int(img_cell_w * 3 / 4), :]
                else:
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[:int(img_cell_h * 3 / 4), int(x_overlap):, :]
            elif k != np.max(col_ids):
                if i == 0:  # 每行的第一张图片取3/4大小
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[int(img_cell_h / 4):int(img_cell_h * 3 / 4), :int(img_cell_w * 3 / 4), :]
                elif i != np.max(row_ids):  # 不是第一张也不是最后一张取1/4到3/4
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[int(img_cell_h / 4):int(img_cell_h * 3 / 4), int(img_cell_w / 4):int(img_cell_w * 3 / 4),
                          :]
                else:
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[int(img_cell_h / 4):int(img_cell_h * 3 / 4), int(x_overlap):, :]
            else:
                if i == 0:
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[int(y_overlap):, :int(img_cell_w * 3 / 4), :]
                elif i != np.max(row_ids):
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[int(y_overlap):, int(img_cell_w / 4):int(img_cell_w * 3 / 4), :]
                else:
                    img = cv2.imread(img_cell_mosaic + str(k) + "_" + str(i) + ".png")
                    img = img[int(y_overlap):, int(x_overlap):, :]
            # print(img.shape)
            row_imgs.append(img)
        result = np.hstack(row_imgs)
        col_imgs.append(result)
    result = np.vstack(col_imgs)
    print(result.shape)

    cv2.imwrite(merge_img_path, result)
    print("done")


def mosaic_img_cell(img_cell_path, merge_img_path):
    file_list = os.listdir(img_cell_path)
    col_ids = []
    row_ids = []
    for file in file_list:
        file_name = os.path.splitext(file)[0]
        col_id = int(file_name.split("_")[0])
        row_id = int(file_name.split("_")[1])
        col_ids.append(col_id)
        row_ids.append(row_id)
    col_imgs = []
    x_overlap, y_overlap = get_overlab(src_path)

    for k in range(len(np.unique(col_ids))):
        row_imgs = []
        for i in range(len(np.unique(row_ids))):
            if k != np.max(col_ids):
                if i != np.max(row_ids):
                    img = cv2.imread(img_cell_path + str(k) + "_" + str(i) + ".png")
                else:
                    img = cv2.imread(img_cell_path + str(k) + "_" + str(i) + ".png")
                    img = img[:, x_overlap:, :]
            else:
                if i != np.max(row_ids):
                    img = cv2.imread(img_cell_path + str(k) + "_" + str(i) + ".png")
                    img = img[y_overlap:, :, :]
                else:
                    img = cv2.imread(img_cell_path + str(k) + "_" + str(i) + ".png")
                    img = img[y_overlap:, x_overlap:, :]
            row_imgs.append(img)
        result = np.hstack(row_imgs)
        col_imgs.append(result)
    result = np.vstack(col_imgs)
    print(result.shape)

    cv2.imwrite(merge_img_path, result)


# 精度报告
def predict_report(pred_img,gt):


    # 预测结果图像
    y_pred = cv2.imread(pred_img, cv2.IMREAD_GRAYSCALE)
    print(y_pred.shape)
    y_pred = y_pred.flatten()

    # 地面真值
    y_true = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
    y_true = y_true.flatten()
    print(y_true.shape)
    print(np.unique(y_pred))
    print(np.unique(y_true))

    # 混淆矩阵
    mat = confusion_matrix(y_true, y_pred)
    print(mat.shape)
    print(mat)
    classify_report = classification_report(y_true, y_pred)
    confusionMatrix = confusion_matrix(y_true, y_pred)
    overall_accuracy = accuracy_score(y_true, y_pred)
    acc_for_each_class = precision_score(y_true, y_pred, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    score = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    # export report
    print('classify_report : \n', classify_report)
    print('confusion_matrix : \n', confusionMatrix)
    print('acc_for_each_class : \n', acc_for_each_class)
    print('average_accuracy(AA): {0:f}'.format(average_accuracy))
    print('overall_accuracy(OA): {0:f}'.format(overall_accuracy))
    print('score: {0:f}'.format(score))
    print("kappa:{0:f}".format(kappa))


src_path = "../datasets/mishan/roi_4.tif"
# src_path = "../datasets/mishan/roi_3_clip.tif"
# 按1/2区域重叠切割图片
# get_img_cell_half(img_cell_w=256, img_cell_h= 256, src_img_path=src_path)

# 按1/2区域重叠拼接
merge_img_path = "../datasets/mishan/pred/mishan_mosaic_gray_deep_xception_0_test.tif"
# 拼接灰度图像
# mosaic_img_cell_2(img_cell_mosaic=img_cell_pred_path, merge_img_path=merge_img_path)

# 拼接彩色图像
merge_img_rgb_path = "../datasets/mishan/pred/mishan_mosaic_rgb_psp_1_roi_4.tif"
mosaic_img_cell_2(img_cell_mosaic=img_cell_pred_path_rgb, merge_img_path=merge_img_rgb_path)

# 精度报告
# pred_img = "../datasets/mishan/pred/mishan_mosaic_gray.tif"
pred_img = merge_img_path
gt = "../datasets/mishan/roi_3_label_new_gray.tif"
# predict_report(pred_img=pred_img, gt=gt)