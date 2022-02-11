#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import re
import time
import cv2
from datetime import datetime
from skimage import measure
import argparse

###### PARAMETER ###############

size=(1024,1024)
extend = 15
batch_size=1
ss = 10 

def get_args():
    parser = argparse.ArgumentParser(description="consensus prediction")
    parser.add_argument('-t', '--target', default='nh0', type=str, help='name of narrowing/erosion and joint')
    parser.add_argument('-ns', '--num_seed', default='6', type=int, help='number of seeds for train-vali partition')
    parser.add_argument('-e', '--epoch', default='10', type=str, help='number of epochs')
    args = parser.parse_args()
    return args

args = get_args()

name_target = args.target
num_seed = args.num_seed
num_epoch = args.epoch

type_score = name_target[0]
type_joint = name_target[1]
index_joint = int(name_target[2:])

if name_target[:2]=='nh':
    scale_score=4.0
    num_joint = 15
    position_all=['LH','RH']
if name_target[:2]=='nf':
    scale_score=4.0
    num_joint = 6
    position_all=['LF','RF']
if name_target[:2]=='eh':
    scale_score=5.0
    num_joint = 16
    position_all=['LH','RH']
if name_target[:2]=='ef':
    scale_score=10.0
    num_joint = 6
    position_all=['LF','RF']

path1='../../data/image/'
path_pred='./pred_' + name_target + '/'
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
dict_erosion_subset={}
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
    if the_individual not in dict_erosion_subset.keys():
        dict_erosion_subset[the_individual]={}
    if the_pos not in dict_erosion_subset[the_individual].keys():
        dict_erosion_subset[the_individual][the_pos]=[]
    dict_erosion_subset[the_individual][the_pos].append([y,x])

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
                y,x = dict_erosion_subset[the_individual][the_pos][k]
            if k>=1 and k<=9:
                y,x = dict_narrow[the_individual][the_pos][k-1]
            if k>=10 and k <=11:
                y,x = dict_erosion_subset[the_individual][the_pos][k-9]
            if k==12:
                y,x = dict_narrow[the_individual][the_pos][14]
            if k>=13 and k <=15:
                y,x = dict_erosion_subset[the_individual][the_pos][k-10]
            dict_erosion[the_individual][the_pos].append([y,x])
#################################

## id partition ##################
individual_test=[]
f=open('individual_test.txt')
for line in f:
    line=line.rstrip()
    individual_test.append(line)

f.close()
individual_test.sort()

id_test=[]
for the_individual in individual_test:
    for the_position in position_all:
        id_test.append(the_individual + '-' + the_position)

###############

if type_score == 'n':
    dict_coordinate = dict_narrow
if type_score == 'e':
    #dict_coordinate = dict_erosion
    dict_coordinate = dict_erosion_subset # for eh, this is dict_erosion_subset instead of dict_erosion

if __name__ == '__main__':

    dist_scaled_all=[]

    for the_id in id_test:
        the_individual, the_pos = the_id.split('-')

        # gt
        center_y = dict_coordinate[the_individual][the_pos][index_joint][0]
        center_x = dict_coordinate[the_individual][the_pos][index_joint][1]

        # pred
        y=[]
        x=[]
        y_all=[]
        x_all=[]
        cutoff1=0.01
        cutoff2=1-cutoff1
        for i in np.arange(num_seed):
            the_path='pred_' + name_target + '_epoch' + num_epoch + '_seed' + str(i) + '/'
            tmp = np.load(the_path + the_id + '.npy')
            y_all.append(tmp[0])
            x_all.append(tmp[1])
            # if at corners, exclude
            if tmp[0] < cutoff1 and tmp[1] < cutoff1:
                continue
            if tmp[0] > cutoff2 and tmp[1] < cutoff1:
                continue
            if tmp[0] < cutoff1 and tmp[1] > cutoff2:
                continue
            if tmp[0] > cutoff2 and tmp[1] > cutoff2:
                continue
            y.append(tmp[0])
            x.append(tmp[1])

        y=np.array(y)
        x=np.array(x)
        y_all=np.array(y_all)
        x_all=np.array(x_all)

        d0 = len(y)

        # if all models fail, then predict 0 ...
        if d0==0:
            y=0
            x=0

        if d0>3:
            mat1=(y.reshape(1,d0) - y.reshape(d0,1))**2
            mat2=(x.reshape(1,d0) - x.reshape(d0,1))**2
            mat = mat1+mat2
            # exclude one potential outlier
            index = np.argmax(np.mean(mat,axis=0))
            x=x[np.arange(d0)!=index]
            y=y[np.arange(d0)!=index]
            mat=mat[np.arange(d0)!=index, :][:, np.arange(d0)!=index]

        pred_center_y = np.median(y)
        pred_center_x = np.median(x)
        pred_coordinate = np.array([pred_center_y, pred_center_x])
        np.save(path_pred + the_id, pred_coordinate)

        # distance to gt
        distance_scaled = ((center_y - pred_center_y)**2 + (center_x - pred_center_x)**2)**0.5
        dist_scaled_all.append(distance_scaled)
        print('%s\t%.4f' % (the_id, distance_scaled))

        ## sanity image #########################
        # image
        image = cv2.imread(path1 + the_id + '.jpg')
        height,width,_ = image.shape
        the_extend = int(extend * (height*width/size[0]/size[1])**0.5)
#        print(image.shape, the_extend)

        # gt
        center_y = int(np.round(height * dict_coordinate[the_individual][the_pos][index_joint][0]))
        center_x = int(np.round(width * dict_coordinate[the_individual][the_pos][index_joint][1]))
        y1 = np.min((center_y + the_extend, height-1))
        x1 = np.max((center_x - the_extend, 0))
        y2 = np.max((center_y - the_extend, 0))
        x2 = np.min((center_x + the_extend, width-1))
        image[y2:y1,x1,2]=255
        image[y2:y1,x2,2]=255
        image[y1,x1:x2,2]=255
        image[y2,x1:x2,2]=255

        # pred
        for i in np.arange(num_seed):
            center_y = int(np.round(height * y_all[i]))
            center_x = int(np.round(width * x_all[i]))
            y1 = np.min((center_y + the_extend, height-1))
            x1 = np.max((center_x - the_extend, 0))
            y2 = np.max((center_y - the_extend, 0))
            x2 = np.min((center_x + the_extend, width-1))
#            print(i, tmp[0], tmp[1],center_y, center_x, y1, x1, y2, x2)
            image[y2:y1,x1,0]=255
            image[y2:y1,x2,0]=255
            image[y1,x1:x2,0]=255
            image[y2,x1:x2,0]=255

        center_y = int(np.round(height * pred_center_y))
        center_x = int(np.round(width * pred_center_x))
        y1 = np.min((center_y + the_extend, height-1))
        x1 = np.max((center_x - the_extend, 0))
        y2 = np.max((center_y - the_extend, 0))
        x2 = np.min((center_x + the_extend, width-1))
        image[y2:y1,x1,1]=255
        image[y2:y1,x2,1]=255
        image[y1,x1:x2,1]=255
        image[y2,x1:x2,1]=255

        cv2.imwrite(path_pred + the_id + '.jpg', image)

        gt_y = dict_coordinate[the_individual][the_pos][index_joint][0]
        gt_x = dict_coordinate[the_individual][the_pos][index_joint][1]
        if distance_scaled > 0.030:
            print(gt_y, gt_x, pred_center_y, pred_center_x)
            print(y_all)
            print(x_all)
        
    dist_scaled_all = np.array(dist_scaled_all)

    print('overall distance > 0.010: n=%d\tpercent=%.2f' % (np.sum(dist_scaled_all>0.010), np.sum(dist_scaled_all>0.010)/len(dist_scaled_all)))
    print('overall distance > 0.015: n=%d\tpercent=%.2f' % (np.sum(dist_scaled_all>0.015), np.sum(dist_scaled_all>0.015)/len(dist_scaled_all)))
    print('overall distance > 0.030: n=%d\tpercent=%.2f' % (np.sum(dist_scaled_all>0.030), np.sum(dist_scaled_all>0.030)/len(dist_scaled_all)))














