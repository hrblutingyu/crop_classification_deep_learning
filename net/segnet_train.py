# coding=utf-8
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import time
from metrics import *
import pickle
from log import *
# 指定GPU训练
import os

# 使用第一张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from SegNet import *
from CASegNet import *
from SASegNet import *
from A2SegNet import *
from generator import *
from plot_metrics import *
import warnings

warnings.simplefilter("ignore")
dataset_path = "../../datasets/WHU-Hi/drSamples/dr_"


# dimensions:#保留的特征数量 = 降维后的通道数
def train(epochs, batch_size, dimensions):
    for k in range(1, 6):
        model = get_CASegNet(256, 256, dimensions)  # 自定义
        saveModelPath = "./trainedModels/"
        model_name = "CA"

        # 保存权重
        weights_filename = saveModelPath + model_name + "_" + str(dimensions) + "_"+str(k) +"_"+ time.strftime("%Y-%m-%d",
                                                                                                    time.localtime()) + ".h5"
        modelcheck = ModelCheckpoint(weights_filename, monitor='val_acc', save_best_only=True, mode='max')
        earlystopping = EarlyStopping(monitor="acc", patience=30, verbose=2, mode='max')
        callable = [modelcheck, earlystopping]

        data_path = dataset_path + str(dimensions) + "/"
        print(data_path)
        train_set_num, val_set_num, _ = get_train_test(data_path)

        print("训练集数量:", train_set_num)
        print("验证集数量:", val_set_num)

        train_data_gene = data_generator(data_path=data_path, batch_size=batch_size, mode="train")
        val_data_gene = data_generator(data_path=data_path, batch_size=batch_size, mode="val")

        start = datetime.now()  # 获得当前时间
        history = model.fit_generator(generator=train_data_gene,
                                      steps_per_epoch=train_set_num // batch_size,
                                      epochs=epochs,
                                      class_weight='auto',
                                      validation_data=val_data_gene,
                                      validation_steps=val_set_num // batch_size,
                                      callbacks=callable,
                                      verbose=1)
        end = datetime.now()  # 获取当前时间
        # 保存日志
        model_log(model_name=model_name, dimensions=dimensions, start=start, end=end, history=history, epochs=epochs)
        # 保存训练过程，方便制图
        history_file_name = saveModelPath + model_name + "_" + str(dimensions) + "_" + str(k) +"_" + time.strftime("%Y-%m-%d",
                                                                                                     time.localtime()) + ".txt"
        with open(history_file_name, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        # 保存模型
        model_filename = saveModelPath + model_name + "_" + str(dimensions) + "_"+ str(k) + "_" + time.strftime("%Y-%m-%d",
                                                                                                  time.localtime()) + ".hdf5"
        model.save(model_filename, overwrite=True)
        # 训练过程可视化
        plotMetrics(history=history, model_name=model_name + "_" + str(dimensions), epochs=epochs)

for i in [20]:
    train(epochs=100, batch_size=4, dimensions=i)
