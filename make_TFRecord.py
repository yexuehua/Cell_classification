#-*-coding:utf-8-*-
import os
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf


IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3

path = "./images/pre_data/test/"


def normal(img):
    max_pixel = np.max(img)
    min_pixel = np.min(img)
    return (img-max_pixel)/(max_pixel-min_pixel)*255

def createTFRecordsFile(src_dir,tfrecords_name):
    dir = src_dir
    writer = tf.python_io.TFRecordWriter(tfrecords_name)

    samples_size = 0
    index = -1
    classes_dict = {}

    for folder_name in os.listdir(dir):
        class_path = dir + folder_name + '/'
        # class_path = dir+'\\'+folder_name+'\\'
        index +=1
        classes_dict[index] = folder_name
        # print(index, folder_name)
        channels = 0
        three_channels = []
        for image_name in os.listdir(class_path):
            channels += 1
            image_path = class_path+image_name
            # print(image_path)
            img_ndarry = cv2.imread(image_path, -1)
            #img_ndarry = normal(img_ndarry)
            # print(img_ndarry,img_ndarry.shape,sep='\n')
            # break
            # height, width = img_ndarry.shape
            img = img_ndarry.flatten()
            # print(img)
            # img = cv_norm(img)
            img = img.tolist()
            # print(type(img),len(img),sep='\n')
            three_channels.append(img)
            # print(type(three_channels),len(three_channels),sep='\n')
            # break
            if channels % 3 == 0:
                single_image = np.array(three_channels)
                single_image = single_image.transpose()
                single_image = np.expand_dims(single_image, axis=0)
                single_images = single_image.reshape((IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS))
                # print(single_images.dtype,single_images.shape,sep='\n')
                # break
                img_raw = single_images.tobytes()
                three_channels.clear()
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            # 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(index)])),
                            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        }
                    )
                )
                writer.write(example.SerializeToString())
                samples_size +=1
        # break
    writer.close()
    print("totally %i samples" %samples_size)
    print(classes_dict)
    return  samples_size,classes_dict


tfrecords_name = "test1_3.tfrecord"
samples_size,classes_dict = createTFRecordsFile(path,tfrecords_name)
print(samples_size,classes_dict,sep='\n')

# test
# samples_size = 234
# classes_dict = {0: 'class_A', 1: 'class_B', 2: 'class_C', 3: 'class_D'}
# train
# samples_size = 702
# classes_dict = {0: 'class_A', 1: 'class_B', 2: 'class_C', 3: 'class_D'}
