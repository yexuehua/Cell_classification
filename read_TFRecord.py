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

# def decodeTFRecordsFile(tfrecords_name):
#     file_queue = tf.train.string_input_producer([tfrecords_name])
#     reader = tf.TFRecordReader()
#     _,serialized_example = reader.read(file_queue)
#     features = tf.parse_single_example(
#         serialized_example,
#         features = {
#             'label':tf.FixedLenFeature([],tf.int64),
#             'image_raw':tf.FixedLenFeature([],tf.string)
#         }
#     )
#     img = tf.decode_raw(features['image_raw'],tf.uint16)
#     img = tf.reshape(img,[IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
#     img = tf.cast(img,tf.float32)*(1./65535)
#     label = tf.cast(features['label'], tf.int32)
#
#
#     return  img,label
#
#
# def input_data(tfrecords_name, batch_size, shuffle = True):
#     image, label = decodeTFRecordsFile(tfrecords_name)
#     if(shuffle):
#         images, labels = tf.train.shuffle_batch([image, label],
#                                                 batch_size=batch_size,
#                                                 capacity=train_samples_size+batch_size,
#                                                 min_after_dequeue=train_samples_size)
#     else:
#         # input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
#         images, labels = tf.train.batch([image,label],
#                                         batch_size=batch_size,
#                                         capacity=batch_size*2)
#     return images,labels

records_queue = tf.train.string_input_producer(['./test.tfrecord'],num_epochs=2)

reader = tf.TFRecordReader()

_,serialized_example = reader.read(records_queue)

features = tf.parse_single_example(
    serialized_example,


    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    }
)
image_raw = features['image_raw']
image = tf.decode_raw(image_raw,tf.int64)
print(image,image.shape,sep='\n')

img = tf.reshape(image,[IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
# img = tf.cast(img,tf.float32)*(1./65535)
label = tf.cast(features['label'], tf.int32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(5):
    print("ggghggghghggh")
    ndarry,lab = sess.run([img,label])
    print("number %d for nadarry===>"%(i+1))
    print(ndarry,ndarry.shape,type(ndarry),sep='\n')
    print('lab = ',lab)
    print(np.max(ndarry),np.min(ndarry),sep='**********')

coord.request_stop()
coord.join(threads)
sess.close()

