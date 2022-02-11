#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import re
import time
import glob
import unet
import tensorflow as tf
import keras
from keras import backend as K
import cv2
import argparse
from datetime import datetime
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
from skimage import measure

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.30 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

###### PARAMETER ###############

size=(1024,1024)
extend = 15
batch_size=1

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-t', '--target', default='nh0', type=str, help='name of narrowing/erosion and joint')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    parser.add_argument('-e', '--epoch', default='10', type=str, help='number of epochs')
    args = parser.parse_args()
    return args

args = get_args()

name_target = args.target
seed_partition = args.seed
num_epoch = args.epoch

type_score = name_target[0]
type_joint = name_target[1]
index_joint = int(name_target[2:])

if name_target[:2]=='nh':
    num_joint = 15
    position_all=['LH','RH']
if name_target[:2]=='nf':
    num_joint = 6
    position_all=['LF','RF']
if name_target[:2]=='eh':
    num_joint = 6
    position_all=['LH','RH']
#if name_target[:2]=='ef':
#    num_joint = 6
#    position_all=['LH','RH']

## model
num_class = 1
num_channel = 3
name_model='./epoch' + num_epoch + '/weights_' + name_target + '_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-4,num_class=num_class,num_channel=num_channel,size=size)
model.load_weights(name_model)
#model.summary()

path0='../../data/image/'
path1='../../data/image_anchored/'
path2='../../data/image_cc/'

path_pred='./pred_' + name_target + '_epoch' + num_epoch + '_seed' + str(seed_partition) + '/'
os.system('mkdir -p ' + path_pred)

## id_all
id_all=[]
f=open('../../data/id_all.txt')
for line in f:
    line=line.rstrip()
    id_all.append(line)

f.close()

## id with flipped labels
id_flipped=[]
f=open('../../data/id_flipped.txt')
for line in f:
    line=line.rstrip()
    id_flipped.append(line)

f.close()

# coordinates of narrowing
dict_narrow={}
f=open('../../data/label_narrow_all.txt')
for line in f:
    line = line.strip()
    y = float(line.split("\"y\": ")[1].split(", \"x\"")[0])
    x = float(line.split("\"x\": ")[1].split("}, \"type\"")[0])
    the_individual , the_pos = line.split('\t')[0].split('/')[1].split('.')[0].split('-')
    the_id = the_individual + '-' + the_pos
    if the_id in id_flipped:
        y=1-y
        x=1-x
    if the_individual not in dict_narrow.keys():
        dict_narrow[the_individual]={}
    if the_pos not in dict_narrow[the_individual].keys():
        dict_narrow[the_individual][the_pos]=[]
    dict_narrow[the_individual][the_pos].append([y,x])

f.close()

# coordinates of erosion
dict_tmp={}
f=open('../../data/label_erosion_all.txt')
for line in f:
    line = line.strip()
    y = float(line.split("\"y\": ")[1].split(", \"x\"")[0])
    x = float(line.split("\"x\": ")[1].split("}, \"type\"")[0])
    the_individual , the_pos = line.split('\t')[0].split('/')[1].split('.')[0].split('-')
    the_id = the_individual + '-' + the_pos
    if the_id in id_flipped:
        y=1-y
        x=1-x
    if the_individual not in dict_tmp.keys():
        dict_tmp[the_individual]={}
    if the_pos not in dict_tmp[the_individual].keys():
        dict_tmp[the_individual][the_pos]=[]
    dict_tmp[the_individual][the_pos].append([y,x])

f.close()

dict_erosion={}
for the_id in id_all:
    the_individual, the_pos = the_id.split('-')
    if the_individual not in dict_erosion.keys():
        dict_erosion[the_individual]={}
    if the_pos not in dict_erosion[the_individual].keys():
        dict_erosion[the_individual][the_pos]=[]
    if 'F' in the_pos:
        for k in np.arange(6):
            y,x = dict_narrow[the_individual][the_pos][k]
            dict_erosion[the_individual][the_pos].append([y,x])
    else:
        for k in np.arange(16):
            if k==0:
                y,x = dict_tmp[the_individual][the_pos][k]
            if k>=1 and k<=9:
                y,x = dict_narrow[the_individual][the_pos][k-1]
            if k>=10 and k <=11:
                y,x = dict_tmp[the_individual][the_pos][k-9]
            if k==12:
                y,x = dict_narrow[the_individual][the_pos][14]
            if k>=13 and k <=15:
                y,x = dict_tmp[the_individual][the_pos][k-10]
            dict_erosion[the_individual][the_pos].append([y,x])
#################################

## id partition ##################
individual_test=[]
f=open('individual_test.txt')
for line in f:
    line=line.rstrip()
    individual_test.append(line)

f.close()

id_test=[]
for the_individual in individual_test:
    for the_position in position_all:
        id_test.append(the_individual + '-' + the_position)

###############

if type_score == 'n':
    dict_coordinate = dict_narrow
if type_score == 'e':
    #dict_coordinate = dict_erosion
    dict_coordinate = dict_tmp # for eh, this is dict_tmp instead of dict_erosion

if __name__ == '__main__':

    label_all=[]
    pred_all=[]
    dist_all=[]
    dist_scaled_all=[]

    for the_id in id_test:
        the_individual, the_pos = the_id.split('-')

        # 1. image
        image0 = np.load(path0 + the_individual + '-' + the_pos + '.npy')
        height,width = image0.shape
        image = np.zeros((height, width, 3),dtype='float32')
        image[:,:,0]=image0
        image[:,:,1] = np.load(path1 + the_individual + '-' + the_pos + '.npy')
        image[:,:,2] = np.load(path2 + the_individual + '-' + the_pos + '.npy')

        image = cv2.resize(image, size)

        if 'R' in the_pos:
            image = cv2.flip(image,1)

        output = model.predict(image.reshape(batch_size,size[0],size[1],num_channel))
#        print(output.shape) # first dimension is batch_size
        output = output.reshape(size[0],size[1],1)

        if 'R' in the_pos:
            output = cv2.flip(output,1).reshape(size[0],size[1],1)

        # for plot
        image1 = cv2.imread(path0 + the_individual + '-' + the_pos + '.jpg')
        image1 = cv2.resize(image1, size)
        image_pred=image1.copy()
        image_pred[:,:,0]=output[:,:,0]*255
        image_pred[:,:,1]=0
        image_pred[:,:,2]=0
        cv2.imwrite(path_pred + the_id + '.jpg', image_pred)

        image_cc = image1.copy()
        #the_extend = int(extend * (height*width/size[0]/size[1])**0.5)
        the_extend = extend
        center_y = int(np.round(size[0] * dict_coordinate[the_individual][the_pos][index_joint][0]))
        center_x = int(np.round(size[1] * dict_coordinate[the_individual][the_pos][index_joint][1]))
        y1 = np.min((center_y + the_extend, size[0]-1))
        x1 = np.max((center_x - the_extend, 0))
        y2 = np.max((center_y - the_extend, 0))
        x2 = np.min((center_x + the_extend, size[1]-1))
        image_cc[y2:y1,x1,2]=255
        image_cc[y2:y1,x2,2]=255
        image_cc[y1,x1:x2,2]=255
        image_cc[y2,x1:x2,2]=255

        pred_bin = np.round(output[:,:,0]*255)
        pred_bin[pred_bin>0] = 1
        connected_component = measure.label(pred_bin, neighbors=8)
        count = np.unique(connected_component, return_counts=True)
        index = np.argsort(count[1])
        if len(index) == 1:
            print(the_id, "no connected component detected")
            #pred_center_y = int(np.round(size[0]/2))
            #pred_center_x = int(np.round(size[1]/2))
            pred_center_y = 0
            pred_center_x = 0
            distance = (size[0]**2 + size[1]**2) ** 0.5
            distance_scaled = 1
        else:
            pred_bin[connected_component == count[0][index[-2]]] = 255
            pred_bin[connected_component != count[0][index[-2]]] = 0
            # draw rectangle
            tmp1, tmp2 = np.where(connected_component == count[0][index[-2]])
            y1 = np.max(tmp1)
            y2 = np.min(tmp1)
            x2 = np.max(tmp2)
            x1 = np.min(tmp2)
            image_cc[y2:y1,x1,0]=255
            image_cc[y2:y1,x2,0]=255
            image_cc[y1,x1:x2,0]=255
            image_cc[y2,x1:x2,0]=255
    
            print(the_id, y1-y2, x2-x1)
            cv2.imwrite(path_pred + the_id + '_cc.jpg', image_cc)

            ## center and distance to gt
            pred_center_y = int(np.round((y1+y2)/2))
            pred_center_x = int(np.round((x1+x2)/2))
            distance = ((center_y - pred_center_y)**2 + (center_x - pred_center_x)**2)**0.5
            distance_scaled = distance / (size[0]**2 + size[1]**2) ** 0.5

        pred_coordinate = np.array([pred_center_y/size[0], pred_center_x/size[1]])
        #pred_coordinate = np.array([pred_center_y, pred_center_x])
        np.save(path_pred + the_id, pred_coordinate)

        print('distance to gt: %.1f\t%.4f' % (distance, distance_scaled))
        dist_all.append(distance)
        dist_scaled_all.append(distance_scaled)

    dist_all = np.array(dist_all)
    dist_scaled_all = np.array(dist_scaled_all)

    print('overall distance > 0.010: n=%d\tpercent=%.2f' % (np.sum(dist_scaled_all>0.010), np.sum(dist_scaled_all>0.010)/len(dist_scaled_all)))
    print('overall distance > 0.015: n=%d\tpercent=%.2f' % (np.sum(dist_scaled_all>0.015), np.sum(dist_scaled_all>0.015)/len(dist_scaled_all)))
    print('overall distance > 0.030: n=%d\tpercent=%.2f' % (np.sum(dist_scaled_all>0.030), np.sum(dist_scaled_all>0.030)/len(dist_scaled_all)))


