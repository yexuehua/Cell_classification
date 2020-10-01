'''
this code used for rename the raw_image
'''
import os

path = "/public/home/xiongzhengsi/PycharmProjects/cell_pheontype/pre_data/class_D/"
dirs = os.listdir(path)

numdir = 0
for dir in dirs:
    next_dir = path + dir + '/'
    numdir+=1
    numfile = 0
    for file in os.listdir(next_dir):
        # print(file)
        oldfile = next_dir + file
        newfile = next_dir + file[:7] + '_' + dir + file[7:]
        os.rename(oldfile,newfile)
        numfile+=1
    print('numdir:%d'%numdir,'numfile:%d'%numfile,sep='===>')



