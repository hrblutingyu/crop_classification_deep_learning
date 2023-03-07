# -*- coding: utf-8 -*-
'''
  @Time    : 2023/2/17 18:28
  @Author  : lutingyu
  @FileName: spectral_curve.py
  @Software: PyCharm
  @Description:
'''
import matplotlib

# matplotlib.use("Agg")  # 保存但不显示绘图

import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']

plt.rcParams["font.size"] = 18  # 该语句解决图像中的“-”负号的乱码问题
# plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def build_curve():
    print()
    soybean = [302, 427, 196, 717, 3455, 4300, 4284, 5013, 1985, 845, 8909, 9018]
    soybean = np.array(soybean) / 10000
    corn =    [411, 636, 357, 801, 3444, 4896, 4688, 5243, 2162, 904, 8600, 8209]
    corn=np.array(corn) / 10000
    rice = [277,357,210,507,2117,3397,3278,3514,1057,477,8709,8976]
    rice=np.array(rice) / 10000
    plt.style.use('default')

    plt.plot(corn, color=(0, 1, 0), label='corn', marker='o', markersize='4')
    plt.plot(soybean, color=(1, 0.5, 0.3), label='soybean',marker='o',markersize='4')
    plt.plot(rice, color=(0.08, 0.4, 0.9), label='rice',marker='o',markersize='4')
    # plt.plot(soybean, color="b", label="Validation loss")
    labels = ["2", "3", "4", "5", "6", "7", "8", "8a", "11", "12", "NDVI", "EVI"]
    plt.xticks(np.arange(0, 12, 1), labels=labels)  # 设置横坐标轴的刻度为 0 到 epochs 的数组
    plt.ylabel("Reflectance/Index value")

    plt.legend(loc="best")
    plt.show()

build_curve()
