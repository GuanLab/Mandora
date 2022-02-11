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

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.45 
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#size=(1024,1024)
extend = 192
size = (extend*2, extend*2)
size_shift = 10
batch_size=20
ss = 10

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-t', '--target', default='nh', type=str, help='name of narrowing/erosion and joint')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    args = parser.parse_args()
    return args

args = get_args()

name_target = args.target
seed_partition = args.seed

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
name_model='weights_' + name_target +'_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-3,num_class=num_class,num_channel=num_channel,size=size)
#model.load_weights(name_model)
model.summary()

path0='../../data/image/'
path1='../../data/image_anchored/'
path2='../../data/image_scale255/'
path3='../../data/image_cc_anchored/'
path4='../../data/image_gaussian/'

#path_mask='../../data/mask_erosion/'
path_mask='../../data/mask_narrow/'

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

# old score
#mat=np.loadtxt('../../data/training.csv',delimiter=',',dtype='str')
mat=np.loadtxt('../../data/training_correct.csv',delimiter=',',dtype='str')

dict_score_old_narrow={}
for the_id in id_all:
    the_individual, the_pos = the_id.split('-')
    # 0. score
    if the_pos=='LH':
        score=mat[mat[:,0]==the_individual,48:63].astype('float').flatten()
    if the_pos=='RH':
        score=mat[mat[:,0]==the_individual,63:78].astype('float').flatten()
    if the_pos=='LF':
        score=mat[mat[:,0]==the_individual,78:84].astype('float').flatten()
    if the_pos=='RF':
        score=mat[mat[:,0]==the_individual,84:90].astype('float').flatten()
    if the_individual not in dict_score_old_narrow.keys():
        dict_score_old_narrow[the_individual]={}
    dict_score_old_narrow[the_individual][the_pos]=score

dict_score_old_erosion={}
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
    if the_individual not in dict_score_old_erosion.keys():
        dict_score_old_erosion[the_individual]={}
    dict_score_old_erosion[the_individual][the_pos]=score

# manual score by arcanum449
mat_nh=np.loadtxt('../../data/score_nh.csv',delimiter=',',dtype='str')
mat_nh[mat_nh=='']='0'
mat_nf=np.loadtxt('../../data/score_nf.csv',delimiter=',',dtype='str')
mat_nf[mat_nf=='']='0'

dict_score_narrow_manual={}
for the_id in id_all:
    the_individual, the_pos = the_id.split('-')
    # 0. score
    if the_pos=='LH':
        score=mat_nh[mat_nh[:,0]==the_individual,1:16].astype('float').flatten()
    if the_pos=='RH':
        score=mat_nh[mat_nh[:,0]==the_individual,16:31].astype('float').flatten()
    if the_pos=='LF':
        score=mat_nf[mat_nf[:,0]==the_individual,1:7].astype('float').flatten()
    if the_pos=='RF':
        score=mat_nf[mat_nf[:,0]==the_individual,7:13].astype('float').flatten()
    if the_individual not in dict_score_narrow_manual.keys():
        dict_score_narrow_manual[the_individual]={}
    dict_score_narrow_manual[the_individual][the_pos]=score

mat_eh=np.loadtxt('../../data/score_eh.csv',delimiter=',',dtype='str')
mat_eh[mat_eh=='']='0'
mat_ef=np.loadtxt('../../data/score_ef.csv',delimiter=',',dtype='str')
mat_ef[mat_ef=='']='0'

dict_score_erosion_manual={}
for the_id in id_all:
    the_individual, the_pos = the_id.split('-')
    # 0. score
    if the_pos=='LH':
        score=mat_eh[mat_eh[:,0]==the_individual,1:17].astype('float').flatten()
    if the_pos=='RH':
        score=mat_eh[mat_eh[:,0]==the_individual,17:33].astype('float').flatten()
    if the_pos=='LF':
        score=mat_ef[mat_ef[:,0]==the_individual,1:7].astype('float').flatten()
    if the_pos=='RF':
        score=mat_ef[mat_ef[:,0]==the_individual,7:13].astype('float').flatten()
    if the_individual not in dict_score_erosion_manual.keys():
        dict_score_erosion_manual[the_individual]={}
    dict_score_erosion_manual[the_individual][the_pos]=score

####################################

## id partition ##################
individual_all=np.loadtxt('seed' + str(seed_partition) + '/individual_unet.txt', dtype='str')

np.random.seed(449) # HERE
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

np.savetxt('id_train_seed' + str(seed_partition) + '.txt', id_train, fmt='%s')
np.savetxt('id_vali_seed' + str(seed_partition) + '.txt', id_vali, fmt='%s')

np.random.seed(datetime.now().microsecond)
np.random.shuffle(id_train)
np.random.shuffle(id_vali)

####################################

## oversampling ###################
if type_score == 'n':
    dict_score = dict_score_narrow
    dict_score_old = dict_score_old_narrow
    dict_score_manual = dict_score_narrow_manual 
    dict_coordinate = dict_narrow
if type_score == 'e':
    dict_score = dict_score_erosion
    dict_score_old = dict_score_old_erosion
    dict_score_manual = dict_score_erosion_manual 
    dict_coordinate = dict_erosion

train_pos=[]
train_neg=[]
for the_id in id_train:
    the_individual, the_pos = the_id.split('-')
    score = dict_score[the_individual][the_pos]
    for k in np.arange(len(score)):
        if score[k] > 0:
            train_pos.append(the_id + '-' + str(k))
        else:
            train_neg.append(the_id + '-' + str(k))

train_pos = np.array(train_pos)
train_neg = np.array(train_neg)

if len(train_neg) > len(train_pos):
    num_diff = len(train_neg) - len(train_pos)
    index = np.random.randint(0, len(train_pos), num_diff)
    train_all = np.concatenate((train_neg, train_pos, train_pos[index]))
else:
    num_diff = len(train_pos) - len(train_neg)
    index = np.random.randint(0, len(train_neg), num_diff)
    train_all = np.concatenate((train_neg, train_pos, train_neg[index]))

vali_pos=[]
vali_neg=[]
for the_id in id_vali:
    the_individual, the_pos = the_id.split('-')
    score = dict_score[the_individual][the_pos]
    for k in np.arange(len(score)):
        if score[k] > 0:
            vali_pos.append(the_id + '-' + str(k))
        else:
            vali_neg.append(the_id + '-' + str(k))

vali_pos = np.array(vali_pos)
vali_neg = np.array(vali_neg)

vali_all = np.concatenate((vali_neg, vali_pos))

# shuffle
np.random.seed(datetime.now().microsecond)
np.random.shuffle(train_all)
np.random.shuffle(vali_all)

# number of pos/neg
print('number of pos: ', len(train_pos))
print('number of neg: ', len(train_neg))
print('number of train: ', len(train_all))

## preload images #######################
dict_image={}
for the_id in id_train+id_vali:
    the_individual, the_pos = the_id.split('-')
    image0 = np.load(path0 + the_individual + '-' + the_pos + '.npy')
    height,width = image0.shape
    image = np.zeros((height, width, 5),dtype='float32')
    image[:,:,0]=image0
    image[:,:,1]=np.load(path1 + the_individual + '-' + the_pos + '.npy')
    image[:,:,2]=np.load(path2 + the_individual + '-' + the_pos + '.npy')
    image[:,:,3]=np.load(path3 + the_individual + '-' + the_pos + '.npy')
    image[:,:,4]=np.load(path4 + the_individual + '-' + the_pos + '.npy')
    dict_image[the_id]=image.copy()

dict_mask={}
for the_id in id_train+id_vali:
    dict_mask[the_id] = np.load(path_mask + the_id + '.npy')

## augmentation parameters ######
if_time=False
max_scale=1.15
min_scale=1
if_mag=True
#if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=True
####################################

def generate_data(ids, batch_size, if_train):

    i=0
    while True:
        image_batch = []
        mask_batch = []
        label_batch = []
        for b in np.arange(batch_size):
            if i == len(ids):
                i=0
                np.random.shuffle(ids)

            the_id = ids[i]
            i += 1

            the_individual, the_pos, index_joint = the_id.split('-')
            index_joint = int(index_joint)

            # 0. score
            score = dict_score[the_individual][the_pos][index_joint]
            score_old = dict_score_old[the_individual][the_pos][index_joint]
            score_manual = dict_score_manual[the_individual][the_pos][index_joint]
            if (if_train==1):
                w1 = 1.0
                w2 = 2.0
                score = (score * w1 + score_manual * w2) / (w1 + w2)

            label = score/scale_score

            # 1. image
            image=dict_image[the_individual + '-' + the_pos]
            mask=dict_mask[the_individual + '-' + the_pos]
            height,width = image.shape[:2]

            # 2. image patch
            center_y = int(np.round(height * dict_coordinate[the_individual][the_pos][index_joint][0]))
            center_x = int(np.round(width * dict_coordinate[the_individual][the_pos][index_joint][1]))

            # random shift as augmentation
            if (if_train==1):
                the_shift = np.random.randint(-size_shift, size_shift+1, 1)[0]
                center_y += the_shift
                the_shift = np.random.randint(-size_shift, size_shift+1, 1)[0]
                center_x += the_shift

            the_extend = int(extend * (height*width/4096/4096)**0.5)
            # ajdust for thumb
            if index_joint==0 and type_joint=='f':
                the_extend = int(the_extend*1.4)
            if index_joint==1 and type_joint=='f':
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
            mask_patch = mask[y2:y1,x1:x2]
            mask_patch = cv2.resize(mask_patch, size).reshape(size[0],size[1],1)

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

            if (if_train==1):
                if if_mag:
                    the_mag = min_mag + (max_mag - min_mag) * np.random.uniform(0,1,1)[0]
                    image_patch = image_patch * the_mag
                if if_flip & np.random.randint(2) == 1:
                    # 0 vertically; 1 horizontally; -1 both
                    #the_seed = np.random.randint(-1,2,1)[0]
                    the_seed = 1
                    image_patch = cv2.flip(image_patch, the_seed)
                    mask_patch = cv2.flip(mask_patch, the_seed).reshape(size[0],size[1],1)

            image_batch.append(image_patch)
            mask_batch.append(mask_patch)
            label_batch.append(label)

        image_batch=np.array(image_batch)
        mask_batch=np.array(mask_batch)
        label_batch=np.array(label_batch)
#        print(image_batch.shape, label_batch.shape)
        yield image_batch, [mask_batch,label_batch]

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(train_all, batch_size,True),
    steps_per_epoch=int(len(train_all) // (batch_size)), nb_epoch=10,
#    steps_per_epoch=10, nb_epoch=1,
    validation_data=generate_data(vali_all, batch_size,False),
    validation_steps=int(len(vali_all) // (batch_size)),
#    validation_steps=10,
    callbacks=callbacks,verbose=1)


