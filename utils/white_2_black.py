# -*- coding: utf-8 -*-
'''
  @Time    : 2022/6/8 7:23
  @Author  : lutingyu
  @FileName: black_2_white.py.py
  @Software: PyCharm
  rgb(0,0,0) to rgb(255,255,255)
'''
import numpy as np
import cv2
import os

def convert_black_white(imgFile,saveAsFile):
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    img=cv2.imread(imgFile,cv2.IMREAD_COLOR)
    black_pixels = np.where(
        (img[:, :, 0] == 0) &
        (img[:, :, 1] == 0) &
        (img[:, :, 2] == 0)
    )
    # set those pixels to white
    img[black_pixels] = [255, 255, 255]
    # newImgName = '../data_sc/predict/segnet_result/rgb/map.tif'
    cv2.imwrite(saveAsFile,img)
# 批量转换,参数为文件夹
def convert_black_white_bat(imgFoler):
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    imgs=os.listdir(imgFoler)
    for img in imgs:
        print(img)
        imgFile=cv2.imread(imgFoler+img,cv2.IMREAD_UNCHANGED)
        print(imgFile.shape)
        black_pixels = np.where(
            (imgFile[:, :, 0] == 0) &
            (imgFile[:, :, 1] == 0) &
            (imgFile[:, :, 2] == 0)
        )
        # set those pixels to white
        imgFile[black_pixels] = [255, 255, 255]
        newImgName =imgFoler+img[:-4]+'_white.tif'
        print(newImgName)
        cv2.imwrite(newImgName,imgFile)

# convert_black_white(imgFile='../data_sc/predict/segnet_result/rgb/result_segnet.tif')
convert_black_white_bat(imgFoler='./temp/')