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
from datetime import datetime
import argparse
print('tf-' + tf.__version__, 'keras-' + keras.__version__)

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.20 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

###### PARAMETER ###############

#size=(1024,1024)
extend = 192
size = (extend*2, extend*2)
batch_size=1
ss = 10

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-t', '--target', default='nh', type=str, help='name of narrowing/erosion and joint')
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

## model
num_class = 1
num_channel = 7
name_model='./epoch' + num_epoch + '/weights_' + name_target + '_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-4,num_class=num_class,num_channel=num_channel,size=size)
model.load_weights(name_model)
#model.summary()

path0='../../data/image/'
path1='../../data/image_anchored/'
path2='../../data/image_scale255/'
path3='../../data/image_cc_anchored/'
path4='../../data/image_gaussian/'

path_mask='../../data/mask_erosion/'

path_pred='./pred_' + name_target + '_epoch' + num_epoch + '_seed' + str(seed_partition) + '/' 
os.system('mkdir -p ' + path_pred)

## id_all
id_all=np.loadtxt('../../data/id_all.txt', dtype='str')
## id with flipped labels
id_flipped=np.loadtxt('../../data/id_flipped.txt', dtype='str')

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

## score
mat=np.loadtxt('../../data_0422/training.csv',delimiter=',',dtype='str')

dict_score_narrow={}
for the_id in id_all:
    the_individual, the_pos = the_id.split('-')
    # 0. score
    if the_pos=='LH':
        score=mat[mat[:,0]==the_individual,48:63].astype('float').flatten()
    if the_pos=='RH':
        score=mat[mat[:,0]==the_individual,63:78].astype('float').flatten()
    if the_pos=='LF':
        #score=mat[mat[:,0]==the_individual,78:84].astype('float').flatten()
        score=mat[mat[:,0]==the_individual,[84,79,80,81,82,83]].astype('float').flatten()
    if the_pos=='RF':
        #score=mat[mat[:,0]==the_individual,84:90].astype('float').flatten()
        score=mat[mat[:,0]==the_individual,[78,85,86,87,88,89]].astype('float').flatten()
    if the_individual not in dict_score_narrow.keys():
        dict_score_narrow[the_individual]={}
    dict_score_narrow[the_individual][the_pos]=score

dict_score_erosion={}
for the_id in id_all:
    the_individual, the_pos = the_id.split('-')
    # 0. score
    if the_pos=='LH':
        score=mat[mat[:,0]==the_individual,4:20].astype('float').flatten()
    if the_pos=='RH':
        score=mat[mat[:,0]==the_individual,20:36].astype('float').flatten()
    if the_pos=='LF':
        score=mat[mat[:,0]==the_individual,36:42].astype('float').flatten()
    if the_pos=='RF':
        score=mat[mat[:,0]==the_individual,42:48].astype('float').flatten()
    if the_individual not in dict_score_erosion.keys():
        dict_score_erosion[the_individual]={}
    dict_score_erosion[the_individual][the_pos]=score

####################################

## id partition
individual_test=np.loadtxt('seed' + str(seed_partition) + '/individual_test.txt', dtype='str')

id_test=[]
for the_individual in individual_test:
    for the_position in position_all:
        id_test.append(the_individual + '-' + the_position)

###############

if type_score == 'n':
    dict_score = dict_score_narrow
    dict_coordinate = dict_narrow
if type_score == 'e':
    dict_score = dict_score_erosion
    dict_coordinate = dict_erosion


if __name__ == '__main__':

    dust_all=[]
    pred_all=[]

    pred_per_id=[]
    score_per_id=[]

    pear_all=[]
    rmse_all=[]

    for the_id in id_test:
        the_individual, the_pos = the_id.split('-')
        # 0. score
        the_dust = dict_score[the_individual][the_pos]

        # 1. image
        image0 = np.load(path0 + the_individual + '-' + the_pos + '.npy')
        height,width = image0.shape
        image = np.zeros((height, width, 5),dtype='float32')
        image[:,:,0]=image0
        image[:,:,1]=np.load(path1 + the_individual + '-' + the_pos + '.npy')
        image[:,:,2]=np.load(path2 + the_individual + '-' + the_pos + '.npy')
        image[:,:,3]=np.load(path3 + the_individual + '-' + the_pos + '.npy')
        image[:,:,4]=np.load(path4 + the_individual + '-' + the_pos + '.npy')

        image_plot = cv2.imread(path0 + the_individual + '-' + the_pos + '.jpg')
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1

        image_mask = image_plot.copy()

        # 2. image patch
        the_pred = []
        height,width,_ = image.shape
        for k in np.arange(len(dict_coordinate[the_individual][the_pos])):
            center_y = int(np.round(height * dict_coordinate[the_individual][the_pos][k][0]))
            center_x = int(np.round(width * dict_coordinate[the_individual][the_pos][k][1]))
            the_extend = int(extend * (height*width/4096/4096)**0.5)
            # ajdust for thumb
            if k==0 and type_joint=='f':
                the_extend = int(the_extend*1.4)
            if k==1 and type_joint=='f':
                the_extend = int(the_extend*1.5)
            # zero locates at top left of an image
            # bottom left
            y1 = np.min((center_y + the_extend, height-1))
            x1 = np.max((center_x - the_extend, 0))
            # top right
            y2 = np.max((center_y - the_extend, 0))
            x2 = np.min((center_x + the_extend, width-1))
            # cut patch
            image_patch = image[y2:y1,x1:x2,:]
            image_patch = cv2.resize(image_patch, size)

            # normalize by patch
            image_patch_normalized = np.zeros((size[0],size[1],2))
            # (i) scale255
            x = image_patch[:,:,0]
            the_max=np.max(x)
            the_min=np.min(x)
            image_patch_normalized[:,:,0] = (x - the_min)/(the_max - the_min) * 255 
            # (ii) Gaussian
            the_avg=np.mean(x)
            the_sd=np.std(x)
            image_patch_normalized[:,:,1] = (x - the_avg) / the_sd * 255 
            # concatenate
            image_patch = np.concatenate((image_patch, image_patch_normalized), axis=2)

            image_patch = image_patch.reshape(batch_size,size[0],size[1],num_channel)

            output_mask,output = model.predict(image_patch)
            output = output.flatten()
            # scale back
            output = output * scale_score
            the_pred.append(output)

            # for plot
            image_plot[y2:y1,x1,2]=255
            image_plot[y2:y1,x2,2]=255
            image_plot[y1,x1:x2,2]=255
            image_plot[y2,x1:x2,2]=255
            # this coordinate is different!
            org1 = (x1-3,y2-3)
            org2 = (x2+3,y2-3)
            # add score text
            image_plot = cv2.putText(image_plot, '%.2f' % output, org1, font,  
                   fontScale, color, thickness, cv2.LINE_AA,bottomLeftOrigin=False) 
            image_plot = cv2.putText(image_plot, '%d' % the_dust[k], org2, font,  
                   fontScale, color, thickness, cv2.LINE_AA,bottomLeftOrigin=False) 

            # plot mask predictions
            output_mask = output_mask.reshape(size[0],size[1])
            # resize back
            output_mask = cv2.resize(output_mask, (x2-x1,y1-y2))
            image_mask[y2:y1,x1:x2,:2]=0
            image_mask[y2:y1,x1:x2,2]=np.round(output_mask * 255).astype('int')

        the_pred = np.array(the_pred).flatten()
        the_dust = np.array(the_dust)

        dust_all = np.concatenate((dust_all, the_dust))
        pred_all = np.concatenate((pred_all, the_pred))

        # score
        the_pear=np.corrcoef(the_dust,the_pred)[0,1]
        the_rmse=mse(the_dust,the_pred)**0.5
        pear_all.append(the_pear)
        rmse_all.append(the_rmse)

        print(the_id, '%.2f %.2f' % (the_pear, the_rmse))
        print(the_dust)
        print(the_pred)

        # average score per id
        score_per_id.append(np.mean(the_dust))
        pred_per_id.append(np.mean(the_pred))

        cv2.imwrite(path_pred + the_id + '_' + '%.2f' % the_rmse + '.jpg', image_plot)
        cv2.imwrite(path_pred + the_id + '_mask.jpg', image_mask)
        np.save(path_pred + the_id, the_pred)

    # cor across joints
    pear_all=np.array(pear_all)
    rmse_all=np.array(rmse_all)

    print('pear: mean=' + '%.2f' % np.nanmean(pear_all) +
        '; median=' + '%.2f' % np.nanmedian(pear_all))
    print('rmse: mean=' + '%.2f' % np.nanmean(rmse_all) +
        '; median=' + '%.2f' % np.nanmedian(rmse_all))


    # cor across ids
    score_per_id=np.array(score_per_id)
    pred_per_id=np.array(pred_per_id)

    print('pear across ids: ' + '%.2f' % np.corrcoef(score_per_id,pred_per_id)[0,1])
    print(score_per_id)
    print(pred_per_id)

    print('overall pear: ' + '%.2f' % np.corrcoef(dust_all,pred_all)[0,1])
    print('overall rmse: ' + '%.2f' % mse(dust_all,pred_all)**0.5)




