#!/usr/bin/env python
import os
import sys
import numpy as np
import re
import lightgbm as lgb
from datetime import datetime
import pickle
import argparse

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

def get_args():
    parser = argparse.ArgumentParser(description="run lgbm prediction")
    parser.add_argument('-nl', '--num_lgbm', default='1', type=int, help='number of lgbm predictions')
    args = parser.parse_args()
    return args

args = get_args()

num_lgbm = args.num_lgbm

individual_test=[]
f=open('individual_test.txt')
for line in f:
    line=line.rstrip()
    individual_test.append(line)

f.close()

num = len(individual_test)
for i in np.arange(1, num_lgbm+1): # TODO fix this 0-9
    the_path = 'pred_lgbm5_seed' + str(i) + '/'
    x=np.load(the_path + 'pred_lgbm_eh.npy')
    the_pred=np.concatenate((x[np.arange(0,num*2,2),:], x[np.arange(1,num*2,2),:]),axis=1)
    x=np.load(the_path + 'pred_lgbm_ef.npy')
    the_pred=np.concatenate((the_pred, \
        np.concatenate((x[np.arange(0,num*2,2),:], x[np.arange(1,num*2,2),:]),axis=1)),axis=1)
    x=np.load(the_path + 'pred_lgbm_nh.npy')
    the_pred=np.concatenate((the_pred, \
        np.concatenate((x[np.arange(0,num*2,2),:], x[np.arange(1,num*2,2),:]),axis=1)),axis=1)
    x=np.load(the_path + 'pred_lgbm_nf.npy')
    the_pred=np.concatenate((the_pred, \
        np.concatenate((x[np.arange(0,num*2,2),:], x[np.arange(1,num*2,2),:]),axis=1)),axis=1)
#    print(the_pred.shape)
    if i==1:
        pred = the_pred.copy()
    else:
        pred += the_pred

pred = pred / num_lgbm

# TODO
sum_erosion = np.sum(pred[:,:44],axis=1).reshape((num,1))
sum_narrow = np.sum(pred[:,44:],axis=1).reshape((num,1))
sum_all = sum_erosion + sum_narrow
pred = np.concatenate((sum_all,sum_erosion,sum_narrow,pred), axis=1)

np.savetxt('predictions_tmp.csv', pred, fmt='%.3f', delimiter=',')


