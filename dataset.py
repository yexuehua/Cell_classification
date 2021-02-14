import os
import cv2
import numpy as np


def get_merge_img(path, file_list, img_size, c, index):
    
    merge_image = np.zeros((img_size, img_size, c))
    if c == 1:
        img_name = file_list[index][0]
        img_path = os.path.join(path,img_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(img_size,img_size))
        cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        clahe_img = clahe.apply(img)
        merge_image[:,:,0] = img/255
    else:
        for i,img_name in enumerate(file_list[index]):
            img_path = os.path.join(path,img_name)
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(img_size,img_size))
            cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            clahe_img = clahe.apply(img)
            merge_image[:,:,i] = img/255
    return merge_image


def read_data_file(data_path, c):
    # read the filenames
    data_path_PC9 = os.path.join(data_path,"PC9")
    data_path_PC9GR = os.path.join(data_path,"PC9GR")
    data_PC9_list = os.listdir(data_path_PC9)
    data_PC9GR_list = os.listdir(data_path_PC9GR)

    # create lists to store filename and label
    data_PC9_2d = [[0] * c for _ in range(len(data_PC9_list) // c)]
    data_PC9_label = [[1,0] for _ in range(len(data_PC9_2d))]
    data_PC9GR_2d = [[0] * c for _ in range(len(data_PC9GR_list) // c)]
    data_PC9GR_label = [[0,1] for _ in range(len(data_PC9GR_2d))]
    # add the filenames of channels to a row
    for i, (pc9_name, pcGR_name) in enumerate(zip(data_PC9_list, data_PC9GR_list)):
        row, column = divmod(i, c)
        data_PC9_2d[row][column] = "PC9/"+pc9_name
        data_PC9GR_2d[row][column] = "PC9GR/"+pcGR_name

    # combine the p and n data
    data_PC9_2d.extend(data_PC9GR_2d)
    data_PC9_label.extend(data_PC9GR_label)

    return data_PC9_2d, data_PC9_label


def get_train_batch(train_path, img_size, channel, batch_size):
    # note that the train_set is a path
    train_set,train_set_label = read_data_file(train_path, channel)
    while 1:
        index = np.random.permutation(np.arange(0,len(train_set)))
        batch_imgs = np.zeros((batch_size, img_size, img_size, channel))
        batch_labels = np.zeros((batch_size,2))
        for i,idx in enumerate(index):
            p = i % batch_size
            batch_imgs[p] = get_merge_img(train_path, train_set, img_size, channel,idx)
            batch_labels[p] = train_set_label[idx]
            if (i+1) % batch_size == 0:
                yield batch_imgs, batch_labels


def get_test_data(test_path, img_size, channel):
    test_set,test_set_label = read_data_file(test_path,channel)
    batch_size = len(test_set)
    imgs = np.zeros((batch_size, img_size, img_size, channel))
    for i in range(batch_size):
        imgs[i] = get_merge_img(test_path, test_set, img_size, channel, i)
    return imgs,test_set_label
