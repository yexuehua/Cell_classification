#-*-coding:utf-8-*-
'''
this code used for read the image,and get five images with single channel,concat as a
four rank tensor with five channels.
'''
import os
import tensorflow as tf
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle

# image normalization
def cv_norm(img):
    norm_img = img.astype(np.float32) / 65535.0
    return norm_img

def cv_unnorm(img):
    unnorm_img = img * 65535.0
    return unnorm_img

def save_data(v,filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

def load_data(filename):
    f = open(filename,'rb')
    v = pickle.load(f)
    f.close()
    return v

def read_image(path):
    allFiles = []
    dirs = os.listdir(path)

    for dir in dirs:
        files = path + dir +'/'
        allFiles.append(files)

    images = []
    for i in range(len(allFiles)):
        channels = 0
        five_channels = []
        for pic in os.listdir(allFiles[i]):
            channels+=1
            pic_path = allFiles[i] + pic
            # print(pic_path)
            img_ndarry = cv2.imread(pic_path, -1)
            height, width = img_ndarry.shape
            img = img_ndarry.flatten()
            img = cv_norm(img)
            img = img.tolist()
            five_channels.append(img)
            # print(img,img.shape,sep='\n')
            if channels % 5 == 0:
                single_image = np.array(five_channels)
                single_image = single_image.transpose()
                single_image = np.expand_dims(single_image,axis=0)
                single_image = single_image.resize((height,width,5))
                img_raw = single_image.tobytes()
                five_channels.clear()

    return images





pathA,pathB,pathC = sys.argv[1:4]
save_path = sys.argv[4] #./predata/allimage_data.txt


if __name__ == '__main__':
    imagesA,imagesB,imagesC = np.array(read_image(pathA)),np.array(read_image(pathB)),np.array(read_image(pathC))
    all_images = np.concatenate((imagesA,imagesB,imagesC),axis=0)
    print(all_images.shape)
    print(imagesA.shape,imagesB.shape,imagesC.shape,sep='\n')
    print(imagesA[:3],imagesB[:3],imagesC[:3],sep='\n\n')
    # print('data saved in : %s' % save_data(all_images, save_path))
