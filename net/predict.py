# -*- coding: utf-8 -*-
'''
  @Time    : 2022/10/17 17:04
  @Author  : lutingyu
  @FileName: predict.py
  @Software: PyCharm
  @Description: 加载模型，对test集进行预测，并给出相关指标
'''
from keras.models import load_model
import tensorflow as tf
from generator import *
from metrics import *
from sklearn.metrics import *
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

colors_dict = [(0, 0, 0), (255, 0, 0), (255, 255, 255), (176, 48, 96), (255, 255, 0), (255, 127, 80),
               (0, 255, 0), (0, 205, 0),
               (0, 130, 0), (127, 255, 212), (160, 32, 240), (216, 191, 216), (0, 0, 255), (0, 0, 139),
               (218, 112, 214), (160, 82, 45), (0, 255, 255), (255, 165, 0),
               (127, 255, 0), (139, 139, 0), (0, 139, 139), (205, 181, 205), (238, 154, 0)]


def results_2_img(test_images_path, test_results_path, test_results_vis_path, results):
    imageList = os.listdir(test_images_path)
    print("生成预测结果......")
    for i, result in enumerate(results):
        img = np.zeros(shape=(img_w, img_h, 3), dtype=np.uint8)
        result = results[i]
        # print(result.shape)
        result = np.argmax(result, axis=2)
        # print(result)
        # print(result.shape)
        # 生成灰度图像
        cv2.imwrite(test_results_path + imageList[i][:-4] + ".png", result)
        # 生成真彩色图像
        B, G, R = cv2.split(img)
        for c in range(len(colors_dict)):
            B[result == c] = colors_dict[c][0]
            G[result == c] = colors_dict[c][1]
            R[result == c] = colors_dict[c][2]

        merged = cv2.merge([B, G, R])
        merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

        # cv2.imshow(str(i), merged)
        # cv2.waitKey(0)
        # print(imageList[i][:-4])
        cv2.imwrite(test_results_vis_path + imageList[i][:-4] + ".png", merged)
    print("done")


def predict_test(model_name, test_images_path, test_results_path, test_results_vis_path):
    model = load_model(model_name, custom_objects={'tf': tf, 'mean_iou': mean_iou, 'dice_coef': dice_coef})
    # 清除之前预测生成的图像
    # 删除文件夹下面的所有文件(只删除文件,不删除文件夹)

    # python删除文件的方法 os.remove(path)path指的是文件的绝对路径,如：

    for i in os.listdir(test_results_path):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = test_results_path + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        os.remove(file_data)
    for i in os.listdir(test_results_vis_path):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = test_results_vis_path + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        os.remove(file_data)
    #
    files = os.listdir(test_images_path)
    for file in files:
        rgb_img = np.zeros(shape=(img_w, img_h, 3), dtype=np.uint8)
        print(file)
        img = np.load(test_images_path + file)
        print(img.shape)
        img = np.expand_dims(img, axis=0)  # (1,256,256,15) 增加一个轴
        result = model.predict(img)
        result = result[0]
        result = np.argmax(result, axis=2)
        print(result.shape)
        cv2.imwrite(test_results_path + file[:-4] + ".png", result)
        B, G, R = cv2.split(rgb_img)
        for c in range(len(colors_dict)):
            B[result == c] = colors_dict[c][0]
            G[result == c] = colors_dict[c][1]
            R[result == c] = colors_dict[c][2]

        merged = cv2.merge([B, G, R])
        merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

        cv2.imwrite(test_results_vis_path + file[:-4] + ".png", merged)


# 精度报告
def predict_report(model, c):
    # 混淆矩阵
    pred_img_path = ""
    order = 5
    if model == "SegNet":
        pred_img_path = "../../datasets/WHU-Hi/predict/SegNet_" + str(c) + "_" + str(order) + "_gray.png"
    elif model == "CA":
        pred_img_path = "../../datasets/WHU-Hi/predict/CA_" + str(c) + "_" + str(order) + "_gray.png"
    elif model == "SA":
        pred_img_path = "../../datasets/WHU-Hi/predict/SA_" + str(c) + "_" + str(order) + "_gray.png"
    else:
        pred_img_path = "../../datasets/WHU-Hi/predict/A2_" + str(c) + "_" + str(order) + "_gray.png"
    if pred_img_path == "":
        print("no model selected! error! error!--------------------------")
        return
    print(pred_img_path)
    y_pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
    print(y_pred.shape)
    y_pred = y_pred.flatten()
    y_true = np.load("../../datasets/WHU-Hi/predict/WHU_Hi_HongHu_gt.npy")
    y_true = y_true.flatten()
    print(y_true.shape)
    print(np.unique(y_pred))
    print(np.unique(y_true))
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


model_name = "./trainedModels/CA_20_5_2022-11-30.hdf5"
num_features = 20
test_images_path = "../../datasets/WHU-Hi/drSamples/clip_" + str(num_features) + "/"
test_results_path = "../../datasets/WHU-Hi/drSamples/pred_" + str(num_features) + "/"
test_results_vis_path = "../../datasets/WHU-Hi/drSamples/visualize_" + str(num_features) + "/"
# predict_test(model_name=model_name, test_images_path=test_images_path, test_results_path=test_results_path, test_results_vis_path=test_results_vis_path)
predict_report("CA", num_features)
