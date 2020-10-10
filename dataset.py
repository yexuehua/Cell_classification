import os
import cv2
import numpy as np


def get_merge_img(path, file_list, img_size, c, index):
    merge_image = np.zeros((img_size, img_size, c))
    for i,img_name in enumerate(file_list[index]):
        img_path = os.path.join(path,img_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(img_path)
        #print(np.max(img),np.min(img))
        merge_image[:,:,i] = img
    return merge_image


def read_data_file(train_path, c):
    # read the filenames
    train_path_PC9 = os.path.join(train_path,"PC9")
    train_path_PC9GR = os.path.join(train_path,"PC9GR")
    train_PC9_list = os.listdir(train_path_PC9)
    train_PC9GR_list = os.listdir(train_path_PC9GR)

    # create lists to store filename and label
    train_PC9_2d = [[0] * c for _ in range(len(train_PC9_list) // c)]
    train_PC9_label = [[1,0] for _ in range(len(train_PC9_2d))]
    train_PC9GR_2d = [[0] * c for _ in range(len(train_PC9GR_list) // c)]
    train_PC9GR_label = [[0,1] for _ in range(len(train_PC9GR_2d))]
    # add the filenames of channels to a row
    for i, (pc9_name, pcGR_name) in enumerate(zip(train_PC9_list, train_PC9GR_list)):
        row, column = divmod(i, 4)
        train_PC9_2d[row][column] = "PC9/"+pc9_name
        train_PC9GR_2d[row][column] = "PC9GR/"+pcGR_name

    # combine the p and n data
    train_PC9_2d.extend(train_PC9GR_2d)
    train_PC9_label.extend(train_PC9GR_label)

    return train_PC9_2d, train_PC9_label

def get_batch_data(train_path, img_size, channel, batch_size):
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
