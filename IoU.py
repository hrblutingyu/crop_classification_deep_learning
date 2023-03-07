# -*- coding: utf-8 -*-
# author： admin
# datetime： 2021/12/16 17:19 
# ide： PyCharm
# https://github.com/bubbliiiing/unet-keras/blob/9adaa621502699693c1c3c01f4e596f1a9e6930e/utils/utils_metrics.py
# 将测试集的预测结果保存到本地，再与标签进行mIoU计算
from keras import backend as K
import cv2
import numpy as np
import tensorflow as tf


