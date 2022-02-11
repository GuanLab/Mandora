#!/usr/bin/env python
import os
import sys
import numpy as np
import re
#import lightgbm as lgb
import sklearn
from sklearn import ensemble
from datetime import datetime
import pickle
import argparse

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()**0.5

np.set_printoptions(precision=3,suppress=True)

###### PARAMETER ###############
num_boost=500
num_early_stop=20
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 5,
    'min_data_in_leaf': 3,
    #'learning_rate': 0.05,
    'verbose': 0,
    'lambda_l2': 2.0,
    'bagging_freq': 1,
    'bagging_fraction': 0.7,
}
################################

def get_args():
    parser = argparse.ArgumentParser(description="run lgbm prediction")
    parser.add_argument('-t', '--target', default='nh', type=str, help='name of narrowing/erosion and joint')
    parser.add_argument('-su', '--seed_unet', default='1', type=int, help='seed of unet')
#    parser.add_argument('-sl', '--seed_lgbm', default='1', type=int, help='seed for lgbm')
    args = parser.parse_args()
    return args

args = get_args()

name_target = args.target
type_score = name_target[0]
type_joint = name_target[1]

if type_joint == 'h':
    position_all=['LH','RH']
    position_other_all=['LF','RF']
    type_other_joint = 'f'
if type_joint == 'f':
    position_all=['LF','RF']
    position_other_all=['LH','RH']
    type_other_joint = 'h'

seed_unet = args.seed_unet
#seed_lgbm = args.seed_lgbm

# score #########################
mat=np.loadtxt('../../data_0422/training.csv',delimiter=',',dtype='str')

id_all=[]
f=open('../../data/id_all.txt')
for line in f:
    line=line.rstrip()
    id_all.append(line)

f.close()

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

#######################################

# id partition ########################
individual_lgbm=np.loadtxt('seed' + str(seed_unet) + '/individual_lgbm.txt', dtype='str')

#np.random.seed(seed_lgbm)
np.random.shuffle(individual_lgbm)
ratio=[0.5,0.5]
num = int(len(individual_lgbm)*ratio[0])
individual_train = individual_lgbm[:num]
individual_vali = individual_lgbm[num:]

id_train=[]
for the_individual in individual_train:
    for the_position in position_all:
        id_train.append(the_individual + '-' + the_position)

id_vali=[]
for the_individual in individual_vali:
    for the_position in position_all:
        id_vali.append(the_individual + '-' + the_position)

individual_test=np.loadtxt('seed' + str(seed_unet) + '/individual_test.txt', dtype='str')

id_test=[]
for the_individual in individual_test:
    for the_position in position_all:
        id_test.append(the_individual + '-' + the_position)

np.random.seed(datetime.now().microsecond)
np.random.shuffle(id_train)
np.random.shuffle(id_vali)
######################################

if name_target == 'nh':
    dict_score = dict_score_narrow
    num_joint = 15
    the_path1 = './pred_patch_nh_seed'
    the_path2 = './pred_patch_nf_seed'
    the_path3 = './pred_patch_eh_seed'
    the_path4 = './pred_patch_ef_seed'
if name_target == 'nf':
    dict_score = dict_score_narrow
    num_joint = 6
    the_path1 = './pred_patch_nf_seed'
    the_path2 = './pred_patch_nh_seed'
    the_path3 = './pred_patch_ef_seed'
    the_path4 = './pred_patch_eh_seed'
if name_target == 'eh':
    dict_score = dict_score_erosion
    num_joint = 16
    the_path1 = './pred_patch_eh_seed'
    the_path2 = './pred_patch_ef_seed'
    the_path3 = './pred_patch_nh_seed'
    the_path4 = './pred_patch_nf_seed'
if name_target == 'ef':
    dict_score = dict_score_erosion
    num_joint = 6
    the_path1 = './pred_patch_ef_seed'
    the_path2 = './pred_patch_eh_seed'
    the_path3 = './pred_patch_nf_seed'
    the_path4 = './pred_patch_nh_seed'

## prepare data #######################
feature_train=[]
label_train=[]
for the_id in id_train:
    the_individual, the_pos = the_id.split('-')
    the_dust = dict_score[the_individual][the_pos]
    if the_pos == position_all[0]:
        the_other_id = the_individual + '-' + position_all[1]
        the_other_id1 = the_individual + '-' + position_other_all[0]
        the_other_id2 = the_individual + '-' + position_other_all[1]
    else:
        the_other_id = the_individual + '-' + position_all[0]
        the_other_id1 = the_individual + '-' + position_other_all[1]
        the_other_id2 = the_individual + '-' + position_other_all[0]
    i = seed_unet
    the_feature = np.concatenate(( \
        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy')))
    feature_train.append(the_feature)
    label_train.append(the_dust)
    the_feature = np.concatenate(( \
        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy')))
    feature_train.append(the_feature)
    label_train.append(the_dust)

feature_train=np.array(feature_train) # row sample; column feature
label_train=np.array(label_train)
print('feature train: ', feature_train.shape)
print('label train: ', label_train.shape)

feature_vali=[]
label_vali=[]
for the_id in id_vali:
    the_individual, the_pos = the_id.split('-')
    the_dust = dict_score[the_individual][the_pos]
    if the_pos == position_all[0]:
        the_other_id = the_individual + '-' + position_all[1]
        the_other_id1 = the_individual + '-' + position_other_all[0]
        the_other_id2 = the_individual + '-' + position_other_all[1]
    else:
        the_other_id = the_individual + '-' + position_all[0]
        the_other_id1 = the_individual + '-' + position_other_all[1]
        the_other_id2 = the_individual + '-' + position_other_all[0]
    i = seed_unet
    the_feature = np.concatenate(( \
        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy')))
    feature_vali.append(the_feature)
    label_vali.append(the_dust)
    the_feature = np.concatenate(( \
        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy')))
    feature_vali.append(the_feature)
    label_vali.append(the_dust)

feature_vali=np.array(feature_vali) # row sample; column feature
label_vali=np.array(label_vali)
print('feature vali: ', feature_vali.shape)
print('label vali: ', label_vali.shape)

feature_test=[]
label_test=[]
for the_id in id_test:
    the_individual, the_pos = the_id.split('-')
    the_dust = dict_score[the_individual][the_pos]
    if the_pos == position_all[0]:
        the_other_id = the_individual + '-' + position_all[1]
        the_other_id1 = the_individual + '-' + position_other_all[0]
        the_other_id2 = the_individual + '-' + position_other_all[1]
    else:
        the_other_id = the_individual + '-' + position_all[0]
        the_other_id1 = the_individual + '-' + position_other_all[1]
        the_other_id2 = the_individual + '-' + position_other_all[0]
    i = seed_unet
    the_feature = np.concatenate(( \
        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy')))
    feature_test.append(the_feature)
    label_test.append(the_dust)
    the_feature = np.concatenate(( \
        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy'), \
        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy')))
    feature_test.append(the_feature)
    label_test.append(the_dust)

feature_test=np.array(feature_test) # row sample; column feature
label_test=np.array(label_test)
print('feature test: ', feature_test.shape)
print('label test: ', label_test.shape)
###############################################

# build model and test #######################
num_tree=500
max_depth=4
X=np.vstack((feature_train, feature_vali))
Y=np.vstack((label_train, label_vali))

os.system('mkdir -p model_etr5_seed' + str(seed_unet))
pred_lgbm = []
for k in np.arange(num_joint):
    print('build model for ' + name_target + str(k))
    the_model=sklearn.ensemble.ExtraTreesRegressor(n_estimators=num_tree, \
        max_depth=max_depth, random_state=0).fit(X,Y[:,k])
    name_model = './model_etr5_seed' + str(seed_unet) + '/etr_' + name_target + str(k) + '.model'
    pickle.dump(the_model, open(name_model, 'wb'))
    the_model=pickle.load(open(name_model, 'rb'))
    the_pred = the_model.predict(feature_test)
    pred_lgbm.append(the_pred)

pred_lgbm=np.array(pred_lgbm).T # row sample; column joint
pred_patch=feature_test[:,:num_joint] 
print('pred lgbm: ', pred_lgbm.shape)
print('pred patch: ', pred_patch.shape)
###############################################

## evaluation #################################
pred_lgbm = np.mean(pred_lgbm.reshape(-1,2,num_joint),axis=1)
pred_patch = np.mean(pred_patch.reshape(-1,2,num_joint),axis=1)
label_test = np.mean(label_test.reshape(-1,2,num_joint),axis=1)

print('label max:')
print(np.max(label_test,axis=0))
print('patch max:')
print(np.max(pred_patch,axis=0))
print('lgbm max:')
print(np.max(pred_lgbm,axis=0))


the_path='./pred_etr5_seed' + str(seed_unet) + '/'
os.system('mkdir -p ' + the_path)

np.save(the_path + 'label_' + name_target, label_test)
np.save(the_path + 'pred_patch_' + name_target, pred_patch.astype('float'))
np.save(the_path + 'pred_lgbm_' + name_target, pred_lgbm)

# avg
avg_train = np.mean(label_train)
avg_vali = np.mean(label_vali)
avg_test = np.mean(label_test)
print('avg: train %.3f vali %.3f test %.3f' % (avg_train, avg_vali, avg_test))

# cor across ids
score_per_id=np.mean(label_test,axis=1)
pred_patch_per_id=np.mean(pred_patch,axis=1)
pred_lgbm_per_id=np.mean(pred_lgbm,axis=1)
print('pear across ids - patch: ' + '%.3f' % np.corrcoef(score_per_id,pred_patch_per_id)[0,1])
print('pear across ids - lgbm : ' + '%.3f' % np.corrcoef(score_per_id,pred_lgbm_per_id)[0,1])

# rmse baseline
baseline_zero = np.zeros(label_test.shape)
baseline_avg = np.zeros(label_test.shape) + avg_test
print('baseline rmse - zero : ' + '%.3f' % rmse(label_test.flatten(),baseline_zero.flatten()))
print('baseline rmse - avg  : ' + '%.3f' % rmse(label_test.flatten(),baseline_avg.flatten()))

# cor/rmse overall
print('overall pear - patch: ' + '%.3f' % np.corrcoef(label_test.flatten(),pred_patch.flatten())[0,1])
print('overall pear - lgbm : ' + '%.3f' % np.corrcoef(label_test.flatten(),pred_lgbm.flatten())[0,1])
print('overall rmse - patch: ' + '%.3f' % rmse(label_test.flatten(),pred_patch.flatten()))
print('overall rmse - lgbm : ' + '%.3f' % rmse(label_test.flatten(),pred_lgbm.flatten()))

# scale by training avg
pred_patch_scaled = pred_patch / np.mean(pred_patch,axis=0) * np.mean(np.concatenate((label_train,label_vali)),axis=0)
print('overall rmse - patch_scaled: ' + '%.3f' % rmse(label_test.flatten(),pred_patch_scaled.flatten()))
pred_lgbm_scaled = pred_lgbm / np.mean(pred_lgbm,axis=0) * np.mean(np.concatenate((label_train,label_vali)),axis=0)
print('overall rmse - lgbm_scaled: ' + '%.3f' % rmse(label_test.flatten(),pred_lgbm_scaled.flatten()))




