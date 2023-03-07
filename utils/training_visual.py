# -*- coding: utf-8 -*-
'''
  @Time    : 2023/1/4 8:09
  @Author  : lutingyu
  @FileName: training_visual.py
  @Software: PyCharm
  @Description: 训练过程可视化
'''

import pickle
import matplotlib.pyplot as plt

with open('./net/trainedModels/unetpp_3_2023-01-03.txt', 'rb') as file_txt:
    history = pickle.load(file_txt)

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.plot(history['loss'])
plt.plot(history['val_loss'],color="darkgray")

plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val", "loss", "val_loss"], loc="best")
plt.show()
