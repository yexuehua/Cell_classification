#-*-coding:utf-8-*-
import cv2
import os

path = "images/pre_data4/train/PC9"

file_list = os.listdir(path)

file_name = os.path.join(path,file_list[1])
img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
print(img)
