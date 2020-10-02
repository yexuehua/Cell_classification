import tensorflow as tf
from tensorflow.layers import AveragePooling2D
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm,flatten
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from dataset import *

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 4

# data_path = './fruits-360/'

def createTFRecordsFile(src_dir,tfrecords_name):
    dir = src_dir
    writer = tf.python_io.TFRecordWriter(tfrecords_name)

    samples_size = 0
    index = -1
    classes_dict = {}

    for folder_name in os.listdir(dir):
        class_path = dir + '/' + folder_name + '/'
        # class_path = dir+'\\'+folder_name+'\\'
        index +=1
        classes_dict[index] = folder_name
        # print(index, folder_name)
        for image_name in os.listdir(class_path):
            image_path = class_path+image_name
            # print(image_path)
            img = Image.open(image_path)
            img = img.resize((IMAGE_HEIGHT,IMAGE_WIDTH))
            img_raw = img.tobytes()
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
    writer.close()
    print("totally %i samples" %samples_size)
    print(classes_dict)
    return  samples_size,classes_dict


def decodeTFRecordsFile(tfrecords_name):
    file_queue = tf.train.string_input_producer([tfrecords_name])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'label':tf.FixedLenFeature([],tf.int64),
            'image_raw':tf.FixedLenFeature([],tf.string)
        }
    )
    img = tf.decode_raw(features['image_raw'],tf.int64)
    img = tf.reshape(img,[IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
    img = tf.cast(img,tf.float32)*(1./10000)
    label = tf.cast(features['label'], tf.int32)

    return  img,label


def inputs(tfrecords_name, batch_size, shuffle = True):
    image, label = decodeTFRecordsFile(tfrecords_name)
    if(shuffle):
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=train_samples_size+batch_size,
                                                min_after_dequeue=train_samples_size)
    else:
        # input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
        images, labels = tf.train.batch([image,label],
                                        batch_size=batch_size,
                                        capacity=batch_size*2)
    return images,labels


def createFruitTrainTFRecordsFile(data_path):
    src_dir = data_path+'Training'
    # src_dir = 'fruits_360_dataset_2018_01_02\Training'
    print('createFruitTrainTFRecordsFile',src_dir)
    tfrecords_name = 'fruits_train.tfrecords'
    samples_size, fruits_dict = createTFRecordsFile(src_dir=src_dir,tfrecords_name=tfrecords_name)
    print('createFruitTrainTFRecordsFile done')
    return samples_size, fruits_dict


def createFruitTestTFRecordsFile(data_path):
    src_dir = data_path+'Validation'
    # src_dir = 'fruits_360_dataset_2018_01_02\Validation'
    print('createFruitTestTFRecordsFile',src_dir)
    tfrecords_name = 'fruits_test.tfrecords'
    samples_size, fruits_dict = createTFRecordsFile(src_dir=src_dir,tfrecords_name=tfrecords_name)
    print('createFruitTestTFRecordsFile done')
    return samples_size, fruits_dict


# train_samples_size, fruits_dict = createFruitTrainTFRecordsFile()#19426
# test_samples_size, fruits_dict = createFruitTestTFRecordsFile()#6523

# for test only
# test_samples_size = 6523
# train_samples_size = 19426
# fruits_dict = {0: 'Apple Red 1', 1: 'Apple Red 2', 2: 'Apple Red 3', 3: 'Apple Red Delicious', 4: 'Apple Red Yellow', 5: 'Apricot',
#                6: 'Avocado', 7: 'Avocado ripe', 8: 'Braeburn', 9: 'Cactus fruit', 10: 'Carambula', 11: 'Cherry', 12: 'Clementine',
#                13: 'Cocos', 14: 'Golden 1', 15: 'Golden 2', 16: 'Golden 3', 17: 'Granadilla', 18: 'Granny Smith', 19: 'Grape Pink',
#                20: 'Grape White', 21: 'Grapefruit', 22: 'Kaki', 23: 'Kiwi', 24: 'Kumquats', 25: 'Lemon', 26: 'Limes', 27: 'Litchi',
#                28: 'Mango', 29: 'Nectarine', 30: 'Orange', 31: 'Papaya', 32: 'Passion Fruit', 33: 'Peach', 34: 'Peach Flat',
#                35: 'Pear', 36: 'Pear Monster', 37: 'Plum', 38: 'Pomegranate', 39: 'Quince', 40: 'Strawberry'}
# ===== flowing senet-inception config
weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1

reduction_ratio = 4

batch_size = 16
iteration = 10
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 1000
# ====== above from senet-inception

train_samples_size = 480
test_samples_size = 120
cell_dict = {0:'PC9',1:'PC9GR'}

# parameters
# learningRate = 0.001
# lr_start = 0.001
# lr_end = 0.0001
# learning_rate = lr_start

# num_steps = 1000
# batch_size = 32
# # update_step = 100
# display_step = 10
# train_acc_target = 1
# train_acc_target_cnt = train_samples_size/batch_size
# if train_acc_target_cnt>20:
#     train_acc_target_cnt = 20

# network parameters
num_input = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS
num_classes = len(cell_dict)
dropout = 0.4

# # saver train parameters
# useCkpt = False
# # checkpoint_step = 5
# # checkpoint_dir = os.getcwd()+'/checkpoint/'
#
# # tf graph input
# X = tf.placeholder(tf.float32,[None,num_input])
# Y = tf.placeholder(tf.float32,[None,num_classes])
# keep_prob = tf.placeholder(tf.float32)


"""
# create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # conv2d wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    # x = tf.nn.conv3d(x,W,strides=[1,strides,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    # return tf.nn.relu(x)
    # return tf.nn.softplus(x)
    return tf.nn.swish(x)

def maxpool2d(x, k=1):
    # max2d wrapper
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def avgpool2d(x):
    return tf.nn.avg_pool(x,ksize=[1,9,9,1],strides=[1,9,9,1],padding='SAME')

def conv_net(X, weights, biases):
    X = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    # convolustion layer
    conv1 = conv2d(X, weights['wc1'], biases['bc1'])
    # max pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=3)

    # convolustion layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # max pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3)

    # apply dropout
    #conv2 = tf.nn.dropout(conv2, 0.3)

    # convolustion layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # max pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=3)

    # apply dropout
    #conv3 = tf.nn.dropout(conv3, 0.3)

    # convolustion layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # max pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=3)

    # apply dropout
    #conv4 = tf.nn.dropout(conv4, 0.3)

    # convolustion layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # max pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2)

    # apply dropout
    #conv5 = tf.nn.dropout(conv5, 0.3)

    shape = conv5.get_shape()
    # fully connected layer
    # fc1 = tf.reshape(conv5, shape=[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.reshape(conv5, shape=[-1, shape[1] * shape[2] * shape[3]])
    # print('conv4 shape:', conv4.shape, ', fc1 shape:', fc1.shape)
    # fc1 = tf.add(tf.matmul(fc1,weights['wd1']), biases['bd1'])
    w = tf.Variable(tf.random_normal([shape[1].value * shape[2].value * shape[3].value, 2048]), dtype=tf.float32)
    fc1 = tf.add(tf.matmul(fc1, w), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # apply dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    # output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# store layers weighta and bias
weights = {
    # 5x5 conv, 3 inputs, 16 outpus
    'wc1': tf.get_variable('wc1',[3,3,4,8],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    # 5x5 conv, 16 input, 32 outpus
    'wc2': tf.get_variable('wc2',[3,3,8,16],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.get_variable('wc3',[3,3,16,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    # 5x5 conv, 64 inputs, 128 outputs
    'wc4': tf.get_variable('wc4',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    # 5x5 conv, 128 inputs, 256 outputs
    'wc5': tf.get_variable('wc5', [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),

    'wc6': tf.get_variable('wc6', [3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'wc7': tf.get_variable('wc7', [3, 3, 32, 4], initializer=tf.contrib.layers.xavier_initializer_conv2d()),

    # fully connected, 7*7*128 inputs, 2048 outputs
    'wd1': tf.get_variable('wd1',[9*9*128,2048],initializer=tf.contrib.layers.xavier_initializer()),
    # 32 inputs, 26 outputs (class prediction)
    'out': tf.get_variable('fc1',[2048,num_classes],initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.Variable(tf.zeros([8])),
    'bc2': tf.Variable(tf.zeros([16])),
    'bc3': tf.Variable(tf.zeros([32])),
    'bc4': tf.Variable(tf.zeros([64])),
    'bc5': tf.Variable(tf.zeros([64])),
    'bc6': tf.Variable(tf.zeros([32])),
    'bc7': tf.Variable(tf.zeros([4])),
    'bd1': tf.Variable(tf.zeros([2048])),
    'out': tf.Variable(tf.zeros([num_classes]))
}
"""


def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding)
        if activation:
            network = Relu(network)
        return network


def Fully_connected(x, units=2, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=True, units=units)


def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x):
    return tf.nn.sigmoid(x)


def Global_Average_Pooling(x):
    x_shape = x.get_shape()
    avg_pool = AveragePooling2D((x_shape[1], x_shape[2]), strides=1)(x)
    avg_pool = tf.reshape(avg_pool,[-1,avg_pool.get_shape()[-1]])
    return avg_pool


def Max_pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Dropout(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        # test_batch_x = test_x[test_pre_index: test_pre_index + add]
        # test_batch_y = test_y[test_pre_index: test_pre_index + add]
        # test_pre_index = test_pre_index + add

        test_batch_x, y = sess.run([test_img, test_label])
        test_batch_y = np.zeros(shape=[batch_size, num_classes])
        for i in range(batch_size):
            test_batch_y[i, y[i]] = 1

        # print("================================================================")
        test_batch_x = np.reshape(test_batch_x, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


class SE_Inception_resnet_v2():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=32, kernel=[3, 3], stride=2, padding='VALID', layer_name=scope + '_conv1')
            x = conv_layer(x, filter=32, kernel=[3, 3], padding='VALID', layer_name=scope + '_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3, 3], layer_name=scope + '_conv3')

            split_max_x = Max_pooling(block_1)
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3, 3], stride=2, padding='VALID',
                                      layer_name=scope + '_split_conv1')
            x = Concatenation([split_max_x, split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3, 3], padding='VALID',
                                       layer_name=scope + '_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1, 1], layer_name=scope + '_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7, 1], layer_name=scope + '_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1, 7], layer_name=scope + '_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3, 3], padding='VALID',
                                       layer_name=scope + '_split_conv7')

            x = Concatenation([split_conv_x1, split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3, 3], stride=2, padding='VALID',
                                      layer_name=scope + '_split_conv8')
            split_max_x = Max_pooling(x)

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_A(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3, 3], layer_name=scope + '_split_conv3')

            split_conv_x3 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3, 3], layer_name=scope + '_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3, 3], layer_name=scope + '_split_conv6')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3])
            x = conv_layer(x, filter=384, kernel=[1, 1], layer_name=scope + '_final_conv1', activation=False)

            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_B(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=128, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1, 7], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=1152, kernel=[1, 1], layer_name=scope + '_final_conv1', activation=False)
            # 1154
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_C(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=2144, kernel=[1, 1], layer_name=scope + '_final_conv2', activation=False)
            # 2048
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope):
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope):
            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3, 3], layer_name=scope + '_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = Global_Average_Pooling(input_x)
            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale

    def Build_SEnet(self, input_x):
        #input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
        # size 32 -> 96
        print(np.shape(input_x))
        # only cifar10 architecture

        x = self.Stem(input_x, scope='stem')

        for i in range(5):
            x = self.Inception_resnet_A(x, scope='Inception_A' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A' + str(i))

        x = self.Reduction_A(x, scope='Reduction_A')

        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')

        for i in range(10):
            x = self.Inception_resnet_B(x, scope='Inception_B' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B' + str(i))

        x = self.Reduction_B(x, scope='Reduction_B')

        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B')

        for i in range(5):
            x = self.Inception_resnet_C(x, scope='Inception_C' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C' + str(i))

        # channel = int(np.shape(x)[-1])
        # x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C')

        x = Global_Average_Pooling(x)
        x = Dropout(x, rate=0.2, training=self.training)
        x = flatten(x)

        x = Fully_connected(x, class_num, layer_name='final_fully_connected')
        return x

# get data from the tfrecord format
train_tfrec_name = 'train1.tfrecord'
test_tfrec_name = "test1.tfrecord"
train_img, train_label= inputs(train_tfrec_name, batch_size, shuffle = True)
test_img, test_label = inputs(test_tfrec_name,batch_size,shuffle=False)

# get data from the numpy



image_size = 512
img_channels = 4
class_num = 2
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_resnet_v2(x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch % 30 == 0 :
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in tqdm(range(1, iteration + 1)):
            batch_x,batch_y = get_bacth_data("image/train",512,4,batch_size)
            # batch_x, y = sess.run([train_img, train_label])
            # print("----here")
            # print(np.shape(batch_x))
            # batch_y = np.zeros(shape=[batch_size, num_classes])
            # for i in range(batch_size):
            #     batch_y[i, y[i]] = 1


            # print("================================================================")
            #batch_x = np.reshape(batch_x, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            # if pre_index + batch_size < 50000:
            #     batch_x = train_x[pre_index: pre_index + batch_size]
            #     batch_y = train_y[pre_index: pre_index + batch_size]
            # else:
            #     batch_x = train_x[pre_index:]
            #     batch_y = train_y[pre_index:]

            # batch_x = data_augmentation(batch_x)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size


        train_loss /= iteration # average loss
        train_acc /= iteration # average accuracy

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_acc, test_loss, test_summary = Evaluate(sess)

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print(line)

        with open('logs.txt', 'a') as f:
            f.write(line)

        saver.save(sess=sess, save_path='./model/Inception_resnet_v2.ckpt')
