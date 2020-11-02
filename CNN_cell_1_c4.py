import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import pandas as pd

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
train_samples_size = 480
test_samples_size = 120
cell_dict = {0:'PC9',1:'PC9GR'}

# parameters
# learningRate = 0.001
# lr_start = 0.001
# lr_end = 0.0001
# learning_rate = lr_start

num_steps = 1000
batch_size = 32
# update_step = 100
display_step = 10
train_acc_target = 1
train_acc_target_cnt = train_samples_size/batch_size
# if train_acc_target_cnt>20:
#     train_acc_target_cnt = 20

# network parameters
num_input = IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS
num_classes = len(cell_dict)
dropout = 0.4

# saver train parameters
useCkpt = False
# checkpoint_step = 5
# checkpoint_dir = os.getcwd()+'/checkpoint/'

# tf graph input
X = tf.placeholder(tf.float32,[None,num_input])
Y = tf.placeholder(tf.float32,[None,num_classes])


weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1

reduction_ratio = 4

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 100

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if activation :
            network = Relu(network)
        return network

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    x_shape = x.get_shape()
    avg_pool = AveragePooling2D((x_shape[1], x_shape[2]), strides=1)(x)
    avg_pool = tf.reshape(avg_pool, [-1, avg_pool.get_shape()[-1]])
    return avg_pool

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

class SE_Inception_resnet_v2():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_conv1')
            x = conv_layer(x, filter=32, kernel=[3,3], padding='VALID', layer_name=scope+'_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3')

            split_max_x = Max_pooling(block_1)
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')
            x = Concatenation([split_max_x,split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7,1], layer_name=scope+'_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1,7], layer_name=scope+'_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_conv_x1,split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv8')
            split_max_x = Max_pooling(x)

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_A(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3,3], layer_name=scope+'_split_conv3')

            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3,3], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3,3], layer_name=scope+'_split_conv6')

            x = Concatenation([split_conv_x1,split_conv_x2,split_conv_x3])
            x = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)

            x = x*0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_B(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1,7], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7,1], layer_name=scope+'_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=1152, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)
            # 1154
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_C(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1,split_conv_x2])
            x = conv_layer(x, filter=2144, kernel=[1,1], layer_name=scope+'_final_conv2', activation=False)
            # 2048
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope) :
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope) :
            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3,3], layer_name=scope+'_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :


            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def Build_SEnet(self, input_x):
        #input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
        # size 32 -> 96
        print(np.shape(input_x))
        # only cifar10 architecture

        input_x = tf.reshape(input_x, shape=[-1, IMAGE_HIGHT, IMAGE_WIDTH,
                                             IMAGE_CHANNELS])
        x = self.Stem(input_x, scope='stem')

        for i in range(5) :
            x = self.Inception_resnet_A(x, scope='Inception_A'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A'+str(i))

        x = self.Reduction_A(x, scope='Reduction_A')
   
        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')

        for i in range(10)  :
            x = self.Inception_resnet_B(x, scope='Inception_B'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B'+str(i))

        x = self.Reduction_B(x, scope='Reduction_B')
        
        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B')

        for i in range(5) :
            x = self.Inception_resnet_C(x, scope='Inception_C'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C'+str(i))
         
            
        # channel = int(np.shape(x)[-1])
        # x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C')
        
        x = Global_Average_Pooling(x)
        x = Dropout(x, rate=0.2, training=self.training)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x
    

global_step = tf.Variable(0, trainable=False)
decaylearning_rate = tf.train.exponential_decay(0.001, global_step,100, 0.9)

# cconstruct model
# logits = conv_net(X,weights,biases,keep_prob)
logits = conv_net(X,weights,biases)
prediction = tf.nn.softmax(logits)


# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=decaylearning_rate)
train_op = optimizer.minimize(loss=loss_op)

# evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# initialization
init = tf.global_variables_initializer()



def trainModel():
    x_point = []
    y_point = []
    acc_point = []
    acc_meet_target_cnt = 0
    tic = time.time()
    for step in range(1, num_steps + 1):
        with tf.Graph().as_default():
            if train_acc_target_cnt <= acc_meet_target_cnt:
                break

            # batch_x, batch_y = train_next_batch(batch_size)
            # test batch

            batch_x, y = sess.run([images, labels])
            batch_y = np.zeros(shape=[batch_size, num_classes])
            for i in range(batch_size):
                batch_y[i, y[i]] = 1

            # print("================================================================")
            batch_x = np.reshape(batch_x, [batch_size, num_input])

            # run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            # if step % update_step == 0 or step == 1:
            #     loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            #
            #     learning_rate = updateLearningRate(acc, lr_start=lr_start)
            #     # L_loss.append(loss)
            #     if train_acc_target <= acc:
            #         acc_meet_target_cnt += 1
            #     else:
            #         acc_meet_target_cnt = 0
            #     # toc = time.time()

            # if step % update_step == 0 or step == 1:
            #
            #     loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            #     learningRate = updateLearningRate(acc, lr_start=learningRate)
            #     if train_acc_target <= acc:
            #         acc_meet_target_cnt += 1
            #     else:
            #         acc_meet_target_cnt = 0
            #     toc = time.time()

            if step % display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                toc = time.time()
                # for loss curve
                x_point.append(step)
                y_point.append(loss)
                acc_point.append(acc)
                print("{:.4f}".format(toc - tic) + "s,", "step " + str(step) + ", minibatch loss = " + \
                      "{:.4f}".format(loss) + ", training accuracy = " + \
                      "{:.4f}".format(acc))

            # if useCkpt:
            #     if step % checkpoint_step == 0 or train_acc_target_cnt <= acc_meet_target_cnt:
            #         # saver.save(sess,checkpoint_dir+'model.ckpt',global_step=step)
            #         saver.save(sess, checkpoint_dir + 'U2OS_model.ckpt')

    return x_point, y_point, acc_point

def testModel(images, labels):
    # calulate the test data sets accuracy
    samples_untest = test_samples_size
    acc_sum = 0
    loss_sum = 0
    test_sample_sum = 0
    while samples_untest > 0:
        with tf.Graph().as_default():
            test_batch_size = batch_size
            # if (test_batch_size > samples_untest):
                # test_batch_size = samples_untest
            # test_images, test_labels = test_next_batch(batch_size=test_batch_size, shuffle=False)

            # test batch
            test_images, y = sess.run([images, labels])
            test_labels = np.zeros(shape=[test_batch_size, num_classes])
            for i in range(test_batch_size):
                test_labels[i, y[i]] = 1

            test_images = np.reshape(test_images, [test_batch_size, num_input])
            acc,loss = sess.run([accuracy,loss_op], feed_dict={X: test_images, Y: test_labels})
            acc_sum += acc * test_batch_size
            loss_sum += loss * test_batch_size
            # print("samples_untest = ", samples_untest, ", acc_current = ", acc)
            samples_untest -= test_batch_size
            test_sample_sum += test_batch_size
    print("Testing accuracy = ", \
          # sess.run(accuracy,feed_dict={X:mnist.test.images/255, Y:mnist.test.labels}))
          acc_sum / test_sample_sum,'testing loss:',loss_sum/test_sample_sum)

def updateLearningRate(acc,lr_start):
    learning_rate_new = lr_start - acc*lr_start*0.9
    return learning_rate_new

# saver = tf.train.Saver()


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
# start training
with tf.Session(config=config) as sess:
    # run the initailizer
    sess.run(init)

    # train batch
    tfrecords_name = 'train1.tfrecord'
    images, labels = inputs(tfrecords_name, batch_size, shuffle = True)
    # create coord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # if useCkpt:
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         pass

    # train the model
    x_point,y_point,acc_point = trainModel()
    print("Optimization finish!")
    df = pd.DataFrame([x_point,y_point,acc_point])
    df.to_csv("PC9_4_data.csv")
    # test batch
    tfrecords_name = 'test1.tfrecord'
    images, labels)

    # close coord
    coord.request_stop()
    coord.join(threads)
    coord2.request_stop()
    coord2.join(threads2)

    # draw loss curve
    plt.title('loss curve')
    plt.xlabel('num_epoch')
    plt.ylabel('loss value')
    plt.plot(x_point,y_point)
    plt.savefig('PC9_loss_4.png')
    plt.clf()
    plt.title('acc curve')
    plt.xlabel('num_epoch')
    plt.ylabel('accuracy value')
    plt.plot(x_point,acc_point)
    plt.savefig("PC9_acc_4.png")

    sess.close()
