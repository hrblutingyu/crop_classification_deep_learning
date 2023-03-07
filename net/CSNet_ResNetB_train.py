# coding=utf-8
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler

# 指定GPU训练
import os

# 使用第一张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FCSNet_ResNetB import *

from generator import *
from plot_metrics import *

def decay_schedule(epoch, lr):
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    # 每20个epoch learning rate 降低0.01倍
    if (epoch % 20 == 0) and (epoch != 0):
        lr = lr * 0.01
    return lr

def train(epochs, batch_size):
    model = FCSNetResNetB(input_shape=(256,256,4))
    model_name = "FCSNetResNetB_NDVI"
    model_filename = model_name + time.strftime("%Y%m%d", time.localtime()) + ".hdf5"
    weights_filename = model_name + time.strftime("%Y%m%d", time.localtime()) + ".h5"
    modelcheck = ModelCheckpoint(weights_filename, monitor='val_acc', save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor="acc", patience=30, verbose=2, mode='max')
    lr_scheduler = LearningRateScheduler(decay_schedule)
    callable = [modelcheck, earlystopping,lr_scheduler]
    # 注意路径，使用NDVI和没有使用NDVI的路径不同
    # dataPath = "../data_sc/" # 3通道
    dataPath="../data_sc_rgb_ndvi/"   # 4通道
    train_set_num, val_set_num, _ = get_train_val(dataPath)

    print("训练集数量:", train_set_num)
    print("验证集数量:", val_set_num)

    train_data_gene = data_generator(data_path=dataPath, batch_size=batch_size, mode="train")
    val_data_gene = data_generator(data_path=dataPath, batch_size=batch_size, mode="val")


    history = model.fit_generator(generator=train_data_gene,
                                  steps_per_epoch=train_set_num // batch_size,
                                  epochs=epochs,
                                  validation_data=val_data_gene,
                                  validation_steps=val_set_num // batch_size,
                                  callbacks=callable,
                                  verbose=1)
    model.save(model_filename, overwrite=True)
    plotMetrics(history=history, model_name=model_name, epochs=epochs)

#epochs 论文中设计值为250，如果训练unet or segnet bs=30是可以的，如果是FCSNet，bs=15 or 10
train(epochs=200, batch_size=15)
