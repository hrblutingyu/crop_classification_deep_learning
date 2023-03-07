# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 256
img_h = 256
data_path = "data/"
image_sets = ['1.png', '2.png', '3.png']
label_sets = ['1_labels.png', '2_labels.png', '3_labels.png']
folder_list = ["train", "val", "test"]


def make_dirs(data_path):
    for folder in folder_list:
        if folder=="test":
            os.makedirs(os.path.join(data_path, folder + "/images"))
            os.makedirs(os.path.join(data_path, folder + "/labels"))
            os.makedirs(os.path.join(data_path, folder + "/visualize"))
            os.makedirs(os.path.join(data_path, folder + "/predict"))
        else:
            os.makedirs(os.path.join(data_path, folder + "/images"))
            os.makedirs(os.path.join(data_path, folder + "/labels"))

# train_set:0.6 val_set:0.2 test_set:0.2
def build_dataset(src_path="data/src", image_num=120000, split=0.6):
    image_each = image_num / len(image_sets)
    print(image_each)
    img_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        img = cv2.imread(os.path.join(src_path, image_sets[i]))
        label = cv2.imread(os.path.join(src_path, label_sets[i]), 0)  # single channel
        x_height, x_width, _ = img.shape
        while count < image_each:
            random_x = random.randint(0, x_width - img_w - 1)
            random_y = random.randint(0, x_height - img_h - 1)
            img_clip = img[random_y: random_y + img_h, random_x: random_x + img_w, :]
            label_clip = label[random_y: random_y + img_h, random_x: random_x + img_w]
            if count < image_each * split:
                img_name=len(os.listdir(os.path.join(data_path, folder_list[0] + "/images")))
                cv2.imwrite((os.path.join(data_path, folder_list[0] + "/images/%d.png" % img_name)),img_clip)
                cv2.imwrite((os.path.join(data_path, folder_list[0] + "/labels/%d.png" % img_name)), label_clip)
            elif (count >= image_each * split) and (count < image_each * (split+0.2)):
                img_name = len(os.listdir(os.path.join(data_path, folder_list[1] + "/images")))
                cv2.imwrite((os.path.join(data_path, folder_list[1] + "/images/%d.png" % img_name)), img_clip)
                cv2.imwrite((os.path.join(data_path, folder_list[1] + "/labels/%d.png" % img_name)), label_clip)
            else:
                img_name = len(os.listdir(os.path.join(data_path, folder_list[2] + "/images")))
                cv2.imwrite((os.path.join(data_path, folder_list[2] + "/images/%d.png" % img_name)), img_clip)
                cv2.imwrite((os.path.join(data_path, folder_list[2] + "/labels/%d.png" % img_name)), label_clip)
                # label_clip*50 可视化灰度图像,类别较多时位数适当减小，label_clip*50<=255
                cv2.imwrite((os.path.join(data_path, folder_list[2] + "/visualize/%d.png" % img_name)), label_clip*50)

            count += 1
            img_count += 1
#
# if __name__ == '__main__':
#     make_dirs("data/")
#     build_dataset()
