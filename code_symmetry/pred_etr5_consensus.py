import numpy as np
import sys
import os

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

num_unet_seed=10

path0='pred_etr5_consensus/'
os.system('mkdir -p ' + path0)
path1='log_consensus/'
os.system('mkdir -p ' + path1)

target_all=['nh','nf','eh','ef','t']

for the_unet_seed in np.arange(num_unet_seed):
    out=open(path1 + 'score_etr5_consensus_seed' + str(the_unet_seed) + '.txt','w')
    out.write('target\tpear\trmse\n')
    for the_target in target_all:
        filename='pred_etr5_seed' + str(the_unet_seed) + '/pred_lgbm_' + the_target + '.npy'
        pred_lgbm = np.load(filename)
        pred_patch = np.load('pred_etr5_seed' + str(the_unet_seed) + '/pred_patch_' + the_target + '.npy')
        # simply adjust by patch
#        pred_lgbm1 = pred_lgbm/np.mean(pred_lgbm,axis=0)*np.mean(pred_patch,axis=0)
#        w1=3.0; w2=1.0
#        pred_lgbm2 = (pred_lgbm * w1 + pred_patch * w2) / (w1 + w2)
#        pred_lgbm2 = pred_lgbm2/np.mean(pred_lgbm2,axis=0)*np.mean(pred_patch,axis=0)
#        pred_lgbm1 = pred_lgbm/np.mean(pred_lgbm,axis=0)*np.mean(pred_patch,axis=0)
#        pred_lgbm2 = pred_patch/np.mean(pred_patch,axis=0)*np.mean(pred_lgbm,axis=0)
        w1=1.0; w2=1.0 # cv 1:1 best
        pred_lgbm1 = (pred_lgbm * w1 + pred_patch * w2) / (w1 + w2)
        pred_lgbm2 = pred_lgbm1/np.mean(pred_lgbm1,axis=0)*np.mean(pred_lgbm,axis=0)
        label = np.load('pred_etr5_seed' + str(the_unet_seed) + '/label_' + the_target + '.npy')
        tmp1=np.corrcoef(label.flatten(),pred_patch.flatten())[0,1]
        tmp2=mse(label, pred_patch)**0.5
        out.write(the_target + '-patch\t%.3f\t%.3f\n' % (tmp1,tmp2))
        tmp1=np.corrcoef(label.flatten(),pred_lgbm.flatten())[0,1]
        tmp2=mse(label, pred_lgbm)**0.5
        out.write(the_target + '-lgbm\t%.3f\t%.3f\n' % (tmp1,tmp2))
        tmp1=np.corrcoef(label.flatten(),pred_lgbm1.flatten())[0,1]
        tmp2=mse(label, pred_lgbm1)**0.5
        out.write(the_target + '-lgbm1\t%.3f\t%.3f\n' % (tmp1,tmp2))
        tmp1=np.corrcoef(label.flatten(),pred_lgbm2.flatten())[0,1]
        tmp2=mse(label, pred_lgbm2)**0.5
        out.write(the_target + '-lgbm2\t%.3f\t%.3f\n' % (tmp1,tmp2))
        np.save(path0 + 'pred_patch_' + the_target + '.npy', pred_patch)    
        np.save(path0 + 'pred_lgbm_' + the_target + '.npy', pred_lgbm)    
        np.save(path0 + 'label_' + the_target + '.npy', label)    
    
    out.close()
