import os
import sys
import numpy as np
import re
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
from datetime import datetime
import unet
import cv2
import argparse

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.30 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size=(1024,1024)
extend = 15 
batch_size=5

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-t', '--target', default='nh0', type=str, help='name of narrowing/erosion and joint')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    args = parser.parse_args()
    return args

args = get_args()

name_target = args.target
seed_partition = args.seed

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
name_model='weights_' + name_target +'_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-3,num_class=num_class,num_channel=num_channel,size=size)
#model.load_weights(name_model)
model.summary()

path0='../../data/image/'
path1='../../data/image_anchored/'
path2='../../data/image_cc/'

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
individual_all=[]
f=open('individual_unet.txt')
for line in f:
    line=line.rstrip()
    individual_all.append(line)

f.close()

np.random.seed(seed_partition) # HERE
np.random.shuffle(individual_all)
ratio=[0.8,0.2]
num = int(len(individual_all)*ratio[0])
individual_train = individual_all[:num]
individual_vali = individual_all[num:]

id_train=[]
for the_individual in individual_train:
    for the_position in position_all:
        id_train.append(the_individual + '-' + the_position)

id_vali=[]
for the_individual in individual_vali:
    for the_position in position_all:
        id_vali.append(the_individual + '-' + the_position)

os.system('mkdir -p misc')
np.savetxt('misc/id_train_' + name_target + '_seed' + str(seed_partition) + '.txt', id_train, fmt='%s')
np.savetxt('misc/id_vali_' + name_target + '_seed' + str(seed_partition) + '.txt', id_vali, fmt='%s')

np.random.seed(datetime.now().microsecond)
np.random.shuffle(id_train)
np.random.shuffle(id_vali)

####################################

## preload images #######################
dict_image={}
for the_id in id_train+id_vali:
    the_individual, the_pos = the_id.split('-')
    image0 = np.load(path0 + the_individual + '-' + the_pos + '.npy')
    height,width = image0.shape
    image = np.zeros((height, width, 3),dtype='float32')
    image[:,:,0]=image0
    image[:,:,1]=np.load(path1 + the_individual + '-' + the_pos + '.npy')
    image[:,:,2]=np.load(path2 + the_individual + '-' + the_pos + '.npy')
    dict_image[the_id]=image.copy()

## augmentation parameters ######
if_time=False
max_scale=1.15
min_scale=1
if_mag=True
#if_mag=False
max_mag=1.15
min_mag=0.9
#if_flip=True
if_flip=False
####################################

if type_score == 'n':
    dict_coordinate = dict_narrow
if type_score == 'e':
    #dict_coordinate = dict_erosion
    dict_coordinate = dict_erosion_subset # for eh, this is dict_erosion_subset instead of dict_erosion

def generate_data(ids, batch_size, if_train):

    i=0
    while True:
        image_batch = []
        label_batch = []
        for b in np.arange(batch_size):
            if i == len(ids):
                i=0
                np.random.shuffle(ids)

            the_id = ids[i]
            i += 1

            the_individual, the_pos = the_id.split('-')

            # 1. image
            image=dict_image[the_individual + '-' + the_pos]

            # 2. binary mask
            height,width,_ = image.shape
            label = np.zeros((height, width, 1))
            the_extend = int(extend * (height*width/size[0]/size[1])**0.5)
            center_y = int(np.round(height * dict_coordinate[the_individual][the_pos][index_joint][0]))
            center_x = int(np.round(width * dict_coordinate[the_individual][the_pos][index_joint][1]))
            # zero locates at top left of an image
            # bottom left
            y1 = np.min((center_y + the_extend, height-1))
            x1 = np.max((center_x - the_extend, 0))
            # top right
            y2 = np.max((center_y - the_extend, 0))
            x2 = np.min((center_x + the_extend, width-1))
            # binary mask
            label[y2:y1,x1:x2,0]=1

            # cut bottom as the augmentation
            if if_train:
                gt_y = dict_coordinate[the_individual][the_pos][index_joint][0]
                cut_y = np.random.uniform(0.85,0.95,1)[0]
                the_cut = int(np.round(height * np.min((np.max((gt_y+0.01, cut_y)),1))))
                image = image[:the_cut,:,:]
                label = label[:the_cut,:,:]

            # resize
            image = cv2.resize(image, size)
#            image = image[:,:,0].reshape(size[0],size[1],1) 
            label = cv2.resize(label, size)
            label = label.reshape(size[0],size[1],1) 

            if 'R' in the_pos:
                image = cv2.flip(image,1)
                label = cv2.flip(label,1).reshape(size[0],size[1],1)

            if if_train:
                if if_mag:
                    the_mag = min_mag + (max_mag - min_mag) * np.random.uniform(0,1,1)[0]
                    image = image * the_mag
                if if_flip & np.random.randint(2) == 1:
                    # 0 vertically; 1 horizontally; -1 both
                    #the_seed = np.random.randint(-1,2,1)[0]
                    the_seed = 1
                    #image = cv2.flip(image, the_seed).reshape(size[0],size[1],1)
                    image = cv2.flip(image, the_seed)
                    label = cv2.flip(label, the_seed).reshape(size[0],size[1],1)

            image_batch.append(image)
            label_batch.append(label)

        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
#        print(image_batch.shape, label_batch.shape)
        yield image_batch, label_batch

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(id_train, batch_size,True),
    steps_per_epoch=int(len(id_train) // (batch_size)), nb_epoch=50,
#    steps_per_epoch=10, nb_epoch=1,
    validation_data=generate_data(id_vali, batch_size,False),
    validation_steps=int(len(id_vali) // (batch_size)),
#    validation_steps=10,
    callbacks=callbacks,verbose=1)


