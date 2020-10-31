import tensorflow as tf
import numpy as np
# from cifar10 import *
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from model import *
from dataset import *
#import pdb


def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    # test_pre_index = 0
    # add = 1000

    for it in range(test_iteration):
        """
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add
        """

        # /for PC9 data
        testdataGene = get_batch_data("images/pre_data4/test", 512, 4, batch_size)
        test_batch_x, test_batch_y = next(testdataGene)
        # for PC9 data

        """

        test_batch_x, y = sess.run([test_img, test_label])
        test_batch_y = np.zeros(shape=[batch_size, num_classes])
        for i in range(batch_size):
            test_batch_y[i, y[i]] = 1

        # print("================================================================")
        test_batch_x = np.reshape(test_batch_x, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        """

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
# get data from cifar10
# train_x, train_y, test_x, test_y = prepare_data()
# train_x, test_x = color_preprocessing(train_x, test_x)
# print('now output:',test_x.shape)
# get data from the numpy


# ===== flowing senet-inception config
weight_decay = 0.0005
momentum = 0.9
init_learning_rate = 0.1
reduction_ratio = 4
batch_size = 16
iteration = 30
# 128 * 391 ~ 50,000
test_iteration = 10
total_epochs = 500
image_size = 512
img_channels = 4
dropout = 0.7
class_num = 2
# ================ config ===============


x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_resnet_v2(x, class_num, reduction_ratio, training=training_flag).model
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
            #pdb.set_trace()
            # /for PC9 data
            dataGene = get_batch_data("images/pre_data4/train",512,4,batch_size)
            batch_x,batch_y = next(dataGene)
            # for PC9 data
            
            #print(batch_x.shape,batch_y.shape)
            # batch_x, y = sess.run([train_img, train_label])
            # print("----here")
            # print(np.shape(batch_x))
            # batch_y = np.zeros(shape=[batch_size, num_classes])
            # for i in range(batch_size):
            #     batch_y[i, y[i]] = 1


            # print("================================================================")
            #batch_x = np.reshape(batch_x, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            # =======for cifar10 dataset ===================
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
        #pdb.set_trace()

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
