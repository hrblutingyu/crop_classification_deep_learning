# -*- coding: utf-8 -*-
# author： admin
# datetime： 2021/12/19 11:41 
# ide： PyCharm
import matplotlib

matplotlib.use("Agg")  # 保存但不显示绘图
import matplotlib.pyplot as plt
import time

plot_path = "plotMetrics/"


def plotMetrics(history, model_name, epochs):
    # print(history.history.keys())
    figure_name = model_name + "_" + time.strftime("%Y%m%d", time.localtime())
    plt.title("Loss on " + model_name)
    plt.plot(history.history["loss"], color="r", label="Training loss")
    plt.plot(history.history["val_loss"], color="b", label="Validation loss")
    plt.xlabel = "Epochs"
    # plt.xticks(np.arange(0, epochs, 100))  # 设置横坐标轴的刻度为 0 到 epochs 的数组
    plt.ylabel("Loss")

    plt.legend(loc="best")
    plt.savefig(plot_path + figure_name + '_loss.png')

    plt.gcf().clear()
    plt.title("Accuracy on " + model_name)
    plt.plot(history.history["acc"], color="r", label="Training accuracy")
    plt.plot(history.history["val_acc"], color="b", label="Validation accuracy")
    plt.xlabel = "Epochs"
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(plot_path + figure_name + '_accuracy.png')

    plt.gcf().clear()
    plt.title("dice_coef on " + model_name)
    plt.plot(history.history["dice_coef"], color="r", label="Training dice")
    plt.plot(history.history["val_dice_coef"], color="b", label="Validation dice")
    plt.xlabel = "Epochs"
    plt.ylabel("dice_coef")
    plt.legend(loc="best")
    plt.savefig(plot_path + figure_name + '_dice_coef.png')
