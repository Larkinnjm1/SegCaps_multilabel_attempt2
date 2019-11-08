# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:23:22 2019

@author: aczd087
"""
import sys

from LungSeg.dataset import Dataset

import json
from collections import OrderedDict
#import tensorflow as tf
import numpy as np
import json


sys.path.append(r"\\nsq024vs\u8\aczd087\MyDocs\SegCaps_multilabel_attempt2\Lung_Segmentation")


with open(r'U:\SegCaps_multilabel_attempt2\aug_dict_prob.json','r') as fb: 
    aug_dict_vals=json.load(fb)


trl_ds=Dataset([256,256],
               [1,1],5,
               r'F:\Biomedical images\Train\sub sample sets\NIFTI_MR_256x256_250_train_segcaps\train\NIFTI_MR_256x256_png_256grey_lvl\t1dual_inphase',
               True,aug_dict_vals)

#Generating datasets for training and validation. 
trl_ds_train=trl_ds.dataset_train()
trl_ds_val=trl_ds.dataset_val()

trl_ds_int_vals=np.array([])
trl_ds_int_train=np.array([])

for i in range(1,50):
    
    #Getting validatoin values for analysis
    trl_ds_train_tmp=trl_ds_train.get_next()
    trl_ds_val_tmp=trl_ds_val.get_next()
    
    trl_ds_int_vals=np.hstack((trl_ds_int_vals,
                               trl_ds_val_tmp['generators']['data'].flatten()))
    
    trl_ds_int_train=np.hstack((trl_ds_int_train,
                               trl_ds_train_tmp['generators']['data'].flatten()))
    
    
    