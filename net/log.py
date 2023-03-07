# -*- coding: utf-8 -*-
'''
  @Time    : 2021/12/31 9:47
  @Author  : lutingyu
  @FileName: log.py
  @Software: PyCharm
'''
import os.path
import time
import math


def model_log(model_name,dimensions, start, end, history, epochs):
    saveModelPath = "./trainedModels/"  #保存模型及日志的路径

    durn = (end - start).seconds  # 两个时间差，并以秒显示出来
    durn = durn / 3600
    start=start.strftime("%Y-%m-%d %H:%M:%S")
    end = end.strftime("%Y-%m-%d %H:%M:%S")

    log_file =saveModelPath + model_name + "_log.txt"

    if os.path.isfile(log_file):
        log_file = open(log_file, "a")
        log_file.write("\n\nmodel_name:" + model_name +  "_"+ str(dimensions) + "--------------------------------------------"
                       "\n\tstart:" + start + "     end:" + end + "   time:" + str(durn) + "hours" +
                       "\n\tepochs:" + str(epochs) +
                       "\n\tloss:" + str(min(history.history["loss"])) + "      val_loss:" + str(min(history.history["val_loss"])) +
                       "\n\tmean_iou:" + str(max(history.history["mean_iou"])) + "      val_mean_iou:" + str(max(history.history["val_mean_iou"])) +
                       "\n\tacc:" + str(max(history.history["acc"])) + "     val_acc:" + str(max(history.history["val_acc"])))
        log_file.close()
    else:
        log_file = open(log_file, "w")
        log_file.write("\nmodel_name:" + model_name +  "_" + str(dimensions) + "--------------------------------------------"
                       "\n\tstart:" + start + "     end:" + end + "   time:" + str(durn) + "hours" +
                       "\n\tepochs:" + str(epochs) +
                       "\n\tloss:" + str(min(history.history["loss"])) + "      val_loss:" + str(min(history.history["val_loss"])) +
                       "\n\tmean_iou:" + str(max(history.history["mean_iou"])) + "      val_mean_iou:" + str(max(history.history["val_mean_iou"])) +
                       "\n\tacc:" +  str(max(history.history["acc"])) + "     val_acc:" + str(max(history.history["val_acc"])))
        log_file.close()
