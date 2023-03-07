# -*- coding: utf-8 -*-
'''
  @Time    : 2022/10/22 9:12
  @Author  : lutingyu
  @FileName: gray_2_rgb.py
  @Software: PyCharm
  @Description:gray image to true color image
'''
import numpy as np
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2

colors_dict = [(1, 1, 1), (1, 250, 2), (255, 127, 80), (22, 100, 250)]

def gray_to_rgb(gray_image_name, rgb_image_path):
    result = cv2.imread(gray_image_name, 0)
    print(np.unique(result))
    img = np.zeros(shape=(result.shape[0], result.shape[1], 3), dtype=np.uint8)

    print(result.shape)
    # 生成真彩色图像
    B, G, R = cv2.split(img)
    for c in range(len(colors_dict)):
        B[result == c] = colors_dict[c][0]
        G[result == c] = colors_dict[c][1]
        R[result == c] = colors_dict[c][2]

    merged = cv2.merge([B, G, R])
    merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
    cv2.imwrite(rgb_image_path, merged)

gray_image_name = "../datasets/mishan/pred/study_area_new_pred_svm_rbf.png"
rgb_image_path = "../datasets/mishan/pred/study_area_new_pred_rgb_svm_rbf.tif"
gray_to_rgb(gray_image_name=gray_image_name, rgb_image_path=rgb_image_path)
