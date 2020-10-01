# -*- coding:utf-8 -*-
'''
this code used for read the image,and get five images with single channel,concat as a
four rank tensor with five channels.
'''
import os
import tensorflow as tf
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

# def cv_norm(img):
#     norm_img = img.astype('float') / 65535.0
#     return norm_img

path = "/public/home/xiongzhengsi/PycharmProjects/cell_pheontype/IXMtest_M22_s9_w5_AE9D9A2D-9640-4FB9-B683-69010D352E16.tif"
# im = Image.open(path)
# width,height = im.size
# data = im.getdata()
# print(type(data))
# data = np.matrix(data,dtype='float')
# new_data = np.reshape(data,(width,height))
# print(type(new_data))
# print(new_data)
# print(new_data.shape)
# tensor = tf.convert_to_tensor(new_data)
#
# with tf.Session() as sess:
#     print(sess.run(tensor))

# path = "/public/home/xiongzhengsi/PycharmProjects/cell_pheontype/rgb.jpeg"
path2 = "/public/home/xiongzhengsi/PycharmProjects/cell_pheontype/rgb.jpeg"
img = mpimg.imread(path)
print(img.shape)
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()
# img = cv2.imread(path2,-1)
# print(img.dtype)
# img = np.expand_dims(img, axis=0)
# img = np.expand_dims(img,axis=2)
# height,width,channel = img.shape
# print(width,height,channel)
# img2 = cv2.imread(path2,-1).flatten()
# img = cv_norm(img)
# img2 = cv_norm(img2)
# image = np.vstack((img,img2))
# print(image)
# print(image.shape)
# print (img)
# cv2.namedWindow('Image',0)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(type(img))
# print (img.shape)
# print (img.dtype)
# print (img.min())
# print (img.max())
# tensor = tf.convert_to_tensor(img)
# tensor2 = tf.convert_to_tensor(img2)
# print(tensor)
# print(type(tensor))
# img_tensor = tf.to_float(tensor,name='ToFloat')
# print(type(img_tensor))
# print(img_tensor)
# img_tensor2 = tf.to_float(tensor2,name='ToFloat')
# # all_tensor = tf.concat()
# with tf.Session() as sess:
#     img_tensor ,img_tensor2 = sess.run([img_tensor,img_tensor2])
#     print(img_tensor,img_tensor2,sep='\n')
#     print(type(img_tensor))




# path = sys.argv[1]
# print(path)
# def read_image(path):
#     allFiles = []
#     dirs = os.listdir(path)
#
#     for dir in dirs:
#         files = path + dir +'/'
#         allFiles.append(files)
#
#     for i in range(len(allFiles)):
#         for pic in os.listdir(allFiles[i]):
#             pic_path = allFiles[i] + pic
#             # print(pic_path)
#             image = Image.open(pic_path)
#             width, height = image.size
#             data = image.getdata()
#             data = np.matrix(data,dtype='float')
#             new_data = np.reshape(data,(width,height))
#             # print(type(new_data))
#             # print(new_data)
#             # print(new_data.shape)
#             image_tensor = tf.convert_to_tensor(new_data)
#             with tf.Session() as sess:
#                 print(sess.run(image_tensor))
#
#
#             return
#         return
#
#
# if __name__ == '__main__':
#     read_image(path)





