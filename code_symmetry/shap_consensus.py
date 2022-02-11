#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import re
from datetime import datetime
import pickle
import argparse

np.set_printoptions(precision=3,suppress=True)

## 86 features predict 1 target; (15nh + 6nf + 16eh + 6ef)*2
## the order of feature:
# lnh - rnh - lnf - rnf - leh - reh - lef - ref
# lnf - rnf - lnh - rnh - lef - ref - leh - reh
# leh - reh - lef - ref - lnh - rnh - lnf - rnf
# lef - ref - leh - reh - lnf - rnf - lnh - rnh

path0 = './shap_value/'
path_out = './shap_consensus/'
os.system('mkdir -p ' + path_out)

num_seed = 10

mat = np.zeros((0,86))

## consensus shap abs average
name_target = 'nh'
for k in range(15):
    shap_value = np.zeros((0,86))
    for i in range(num_seed):
        shap_value = np.vstack((shap_value, np.load(path0 + name_target + str(k) + '_seed' + str(i) + '.npy')))
    shap_abs = np.abs(shap_value)
    shap_abs_avg = np.mean(shap_abs,axis=0)    
    np.save(path_out + name_target + str(k), shap_abs_avg)
    mat = np.vstack((mat, shap_abs_avg))

name_target = 'nf'
for k in range(6):
    shap_value = np.zeros((0,86))
    for i in range(num_seed):
        shap_value = np.vstack((shap_value, np.load(path0 + name_target + str(k) + '_seed' + str(i) + '.npy')))
    shap_abs = np.abs(shap_value)
    shap_abs_avg = np.mean(shap_abs,axis=0)    
    np.save(path_out + name_target + str(k), shap_abs_avg)
    mat = np.vstack((mat, shap_abs_avg))

name_target = 'eh'
for k in range(16):
    shap_value = np.zeros((0,86))
    for i in range(num_seed):
        shap_value = np.vstack((shap_value, np.load(path0 + name_target + str(k) + '_seed' + str(i) + '.npy')))
    shap_abs = np.abs(shap_value)
    shap_abs_avg = np.mean(shap_abs,axis=0)    
    np.save(path_out + name_target + str(k), shap_abs_avg)
    mat = np.vstack((mat, shap_abs_avg))

name_target = 'ef'
for k in range(6):
    shap_value = np.zeros((0,86))
    for i in range(num_seed):
        shap_value = np.vstack((shap_value, np.load(path0 + name_target + str(k) + '_seed' + str(i) + '.npy')))
    shap_abs = np.abs(shap_value)
    shap_abs_avg = np.mean(shap_abs,axis=0)
    np.save(path_out + name_target + str(k), shap_abs_avg)
    mat = np.vstack((mat, shap_abs_avg))


## convert np array to pd dataframe
df = pd.DataFrame(mat, index = ['nh' + str(i) for i in range(15)] + ['nf' + str(i) for i in range(6)] + \
    ['eh' + str(i) for i in range(16)] + ['ef' + str(i) for i in range(6)])

df.to_csv(path_out + 'shap_consensus.tsv', sep='\t')


