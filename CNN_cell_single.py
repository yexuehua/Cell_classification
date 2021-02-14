import tensorflow as tf
import numpy as np
from sklearn import metrics
# from cifar10 import *
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from model import *
from dataset import *
import matplotlib.pyplot as plt
#import pdb


def Evaluate(sess):
    """
    test_batch_x = test_x[test_pre_index: test_pre_index + add]
    test_batch_y = test_y[test_pre_index: test_pre_index + add]
    test_pre_index = test_pre_index + add
    """

    # /for PC9 data
    test_batch_x, test_batch_y = get_test_data(test_data_path, 512, 1)

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
    sess.run(tf.local_variables_initializer())
    test_pred, test_label = sess.run([pred, data_label], feed_dict = test_feed_dict)
    test_loss, test_acc, test_recall, test_precision, test_TP, test_TN,test_FP,test_FN, test_AUC= sess.run([cost, accuracy, recall, precision, TP, TN, FP,FN, AUC], feed_dict=test_feed_dict)


    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, test_recall, test_precision, test_TP, test_TN, test_FP, test_FN, test_AUC, test_pred, test_label, summary

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
iteration = 12
# 128 * 391 ~ 50,00
total_epochs = 300
image_size = 512
img_channels = 1
dropout = 0.7
class_num = 2
# ================ config ===============
train_data_path = "/public/home/yexuehua/PycharmProjects/Cell_classification/images/pre_data_chondriosome/train"
test_data_path = "/public/home/yexuehua/PycharmProjects/Cell_classification/images/pre_data_chondriosome/test"
result_path = "./result/single_channel"
# training set: 240
# test set: 60

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_resnet_v2(x, class_num, reduction_ratio, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,
#                                   beta2=0.999, epsilon=1e-08,
#                                   use_locking=False,name="Adam")
train = optimizer.minimize(cost + l2_loss * weight_decay)
pred = tf.argmax(logits, 1)
data_label = tf.argmax(label, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
_, recall = tf.metrics.recall(tf.argmax(label,1), tf.argmax(logits,1))
_, precision = tf.metrics.precision(tf.argmax(label,1), tf.argmax(logits,1))
_, TP = tf.metrics.true_positives(tf.argmax(label,1), tf.argmax(logits,1))
_, TN = tf.metrics.true_negatives(tf.argmax(label,1), tf.argmax(logits,1))
_, FP = tf.metrics.false_positives(tf.argmax(label,1), tf.argmax(logits,1))
_, FN = tf.metrics.false_negatives(tf.argmax(label,1), tf.argmax(logits,1))
_, AUC = tf.metrics.auc(tf.argmax(label,1), tf.argmax(logits,1))

saver = tf.train.Saver(tf.global_variables())


with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(result_path + '/model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(result_path + '/logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    x_point = [0]*total_epochs
    y_train_loss = [0]*total_epochs
    y_train_acc =  [0]*total_epochs
    y_test_loss = [0]*total_epochs
    y_test_acc = [0]*total_epochs
    for epoch in range(1, total_epochs + 1):
        if epoch % 30 == 0 :
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in tqdm(range(1, iteration + 1)):
            #pdb.set_trace()
            # /for PC9 data
            dataGene = get_train_batch(train_data_path,512,img_channels,batch_size)
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

        test_acc, test_loss, test_recall, test_precision, test_TP, test_TN, test_FP, test_FN, test_AUC, test_pred, test_label, test_summary = Evaluate(sess)

        # draw ROI
        fpr, tpr, threshold = metrics.roc_curve(test_label, test_pred)
        plt.title("ROC curve")
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % test_AUC)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(result_path + "/" + str(epoch)+"ROC.png")
        plt.clf()

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f,test_loss:%.4f, test_acc: %.4f,test_recall: %.4f, test_precision: %.4f\n" % (epoch, total_epochs, train_loss, train_acc, test_loss,test_acc, test_recall, test_precision)
        print(line)

        with open(result_path + '/xian_logs.txt', 'a') as f:
            f.write(line)

        saver.save(sess=sess, save_path=result_path + '/model/xian_Inception_resnet_v2.ckpt')
        x_point[epoch-1] = epoch-1
        y_train_loss[epoch-1] = train_loss
        y_train_acc[epoch-1] = train_acc
        y_test_acc[epoch-1] = test_acc
        y_test_loss[epoch-1] = test_loss
    df=pd.DataFrame(columns=["x","y_tain_loss","y_train_acc","y_test_acc","TP","TN","FP","FN","AUC"])
    df["x"] = x_point
    df["y_train_loss"] = y_train_loss
    df["y_train_acc"] = y_train_acc
    df["y_test_loss"] = y_test_loss
    df["y_test_acc"] = y_test_acc
    df["TP"] = test_TP
    df["TN"] = test_TN
    df["FP"] = test_FP
    df["FN"] = test_FN
    df["AUC"] = test_AUC

    df.to_csv(result_path+"/xian_result_point.csv")
    # draw loss curve
    plt.title('train_loss_curve')
    plt.xlabel('num_epoch')
    plt.ylabel('loss value')
    plt.plot(x_point,y_train_loss)
    plt.savefig(result_path+'/xian_PC9_train_loss_1.png')
    plt.clf()
    plt.title('train_acc_curve')
    plt.xlabel('num_epoch')
    plt.ylabel('accuracy value')
    plt.plot(x_point,y_train_acc)
    plt.savefig(result_path + "/xian_PC9_train_acc_4.png")
    plt.clf()
    plt.title('test_acc_curve')
    plt.xlabel('num_epoch')
    plt.ylabel('accuracy value')
    plt.plot(x_point,y_test_acc)
    plt.savefig(result_path + "/xian_PC9_test_acc_4.png")
    plt.clf()
    plt.title('test_loss_curve')
    plt.xlabel('num_epoch')
    plt.ylabel('loss value')
    plt.plot(x_point,y_train_loss)
    plt.savefig(result_path + "/xian_PC9_test_loss_4.png")
