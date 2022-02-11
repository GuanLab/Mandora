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
import shap

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()**0.5

np.set_printoptions(precision=3,suppress=True)

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
individual_test=np.loadtxt('seed' + str(seed_unet) + '/individual_test.txt', dtype='str')

id_test=[]
for the_individual in individual_test:
    for the_position in position_all:
        id_test.append(the_individual + '-' + the_position)

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
#    the_feature = np.concatenate(( \
#        np.load(the_path1 + str(i) + '/' + the_id + '.npy'), \
#        np.load(the_path1 + str(i) + '/' + the_other_id + '.npy'), \
#        np.load(the_path2 + str(i) + '/' + the_other_id2 + '.npy'), \
#        np.load(the_path2 + str(i) + '/' + the_other_id1 + '.npy'), \
#        np.load(the_path3 + str(i) + '/' + the_id + '.npy'), \
#        np.load(the_path3 + str(i) + '/' + the_other_id + '.npy'), \
#        np.load(the_path4 + str(i) + '/' + the_other_id2 + '.npy'), \
#        np.load(the_path4 + str(i) + '/' + the_other_id1 + '.npy')))
#    feature_test.append(the_feature)
#    label_test.append(the_dust)

feature_test=np.array(feature_test) # row sample; column feature
label_test=np.array(label_test)
print('feature test: ', feature_test.shape)
print('label test: ', label_test.shape)
###############################################

path_out = './shap_value/'
os.system('mkdir -p ' + path_out)

## shap analysis
for k in np.arange(num_joint):
    print('shap analysis for ' + name_target + str(k))
    filename = './model_etr5_seed' + str(seed_unet) + '/etr_' + name_target + str(k) + '.model'
    the_model=pickle.load(open(filename, 'rb'))
    explainer = shap.TreeExplainer(the_model)
    shap_value = explainer.shap_values(feature_test)
    np.save(path_out + name_target + str(k) + '_seed' + str(seed_unet), shap_value)
    #shap_abs = np.abs(shap_value)
    #shap_abs_avg = np.mean(shap_abs,axis=0)    



