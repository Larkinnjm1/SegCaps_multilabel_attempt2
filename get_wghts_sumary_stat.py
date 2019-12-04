# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:12:09 2019

@author: aczd087
"""

import argparse
import pickle 
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import pandas as pd 
import numpy as np
import os
import pathlib
import ipdb
from scipy import stats

from copy import deepcopy
def parse_arg():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-modl_dir','--model_dir',required=True,type=str,
                        help='directory os walk will run through to find model files')
    parser.add_argument('-w_b_dir','--wght_bias_dir',required=False,type=str,
                        help='file directory where weight and bias shape dictionaries are present to reshape the linear capsule into their internal shapes.')
    parser.add_argument('-o','--output_dir',required=True,type=str,
                        help='output directory to write final files to')
    parser.add_argument('-ky_subdir','--keyword_subdir',required=False,default='weights',type=str,
                        help='keyword that needs to be found in the subdirectory to process forward for analysis')
    
    parser.add_argument('-caps_w_chnl','--capsule_wght_channel',required=False,default=3,type=int,
                        help='index slice where capsule channel is present')
    parser.add_argument('-caps_b_chnl','--capsule_bias_channel',required=False,default=2,type=int,
                        help='index slice where capsule channel is present')
    
    parser.add_argument('-no_per_epoch','--sample_per_epoch',required=False,default=250,type=int,
                        help='number of samples per epoch to divide results by to get average')
    #ipdb.set_trace()
    return parser.parse_args()


def main(args):
    """main wrapper function to perform all actiities within"""
    
    w_b_dict=gen_wght_bias_dict(args)
    
    for idx,(root,subdir,files) in enumerate(os.walk(args.model_dir)):
        #ipdb.set_trace()
        #confirm weights in subdirectory
        if (args.keyword_subdir in subdir):
            #gen final weight directory 
            tmp_wght_dir=os.path.join(root,args.keyword_subdir)
            #Generating temporary path to pull file address list from. 
            tmp_paths=list(pathlib.Path(tmp_wght_dir).rglob('*model*'))
            #String and integer version of final values 
            tmp_epoch_dict_dir=get_epoch_dict_raw(tmp_paths,args)
            #Gett  per array dictionary for file
            tmp_epoch_dict_wght=eval_weights_dict(tmp_epoch_dict_dir)
            
            #Getting final dictionary comprehension of the evaluation of the model substring
            tmp_epoch_reshape,tmp_epoch_summary=gen_weight_file(tmp_epoch_dict_wght,w_b_dict,args)
            file_nm=os.path.basename(root)
            
            
            write_to_output(file_nm,args,
                            reshape=tmp_epoch_reshape,
                            summary=tmp_epoch_summary)
            
def nested_dict_copy(dict_val:dict)->dict:
    """The purpose of this method is to strip values fields and replace with none"""
    
    for k,v in dict_val.items():
        #RECURSION
        if isinstance(v,dict):
            nested_dict_copy(v)
        else:
            dict_val[k]=None
    return dict_val            

def gen_per_cap_stat(tmp_arr:np.ndarray)->dict:
    """The purpose of this method is to acquire the per capsule summary statistics"""
    #ipdb.set_trace()
    tmp_var=stats.describe(tmp_arr)
    #Analysis of temporary array for method 
    tmp_dict={ k:tmp_var.__getattribute__(k) for k in dir(tmp_var) if not k.startswith('_') }
    
    del tmp_dict['count']
    del tmp_dict['index']
    
    tmp_dict['median']=np.median(tmp_arr)
    tmp_dict['count']=np.count_nonzero(tmp_arr)
    
    return tmp_dict
    
def gen_weight_file(tmp_epoch_dict,w_b_dict,args):
    """The purpose of this method is to take file dictionary values and and index slice on per capsule basis to per capsule
    descriptive statistics to see evoluiton of class imbalance during training. """
        #Generating copy of empty dictoinary to write summary information to. 
    #ipdb.set_trace()
    #Parsing dictoinary with new blank dictionary
    summ_dict=deepcopy(tmp_epoch_dict) 
    summ_dict=nested_dict_copy(summ_dict)
    

    for epch_no in list(tmp_epoch_dict.keys()):
        #Retained as separate variable to keep memory space down duyring processing so dictoinary files can be deleted. 
        cap_net_layer_info=tmp_epoch_dict[epch_no]
        i=0
        for cap_layer in list(cap_net_layer_info.keys()):
            #resolved for dictionary deletion during procesing. 
            cap_array=cap_net_layer_info[cap_layer]
            #pulling correct capsule shape based on file name. 
            tmp_array_shp=get_layr_shp(cap_layer,w_b_dict)
            #reshaping array to match final value
            if len(tmp_array_shp)>0:
                tmp_lyr_nm,tmp_lyr_shp=tmp_array_shp[0]
                
                #reshape array as per original file reshap
                #ipdb.set_trace()
                try:
                    b_w_str,chnl_det=det_bias_wght_lyr(cap_layer,args)
                    #Getting number of capsules in layer
                    no_capsules=tmp_lyr_shp[1]
                    if b_w_str=='capsule_wght_channel':
                        #ipdb.set_trace()
                        tmp_reshp_w=np.copy(tmp_lyr_shp)
                        #multiplying set up to apprpriate values multiplying num atoms by num of capsules
                        tmp_reshp_w[-1]=tmp_reshp_w[-1]*tmp_reshp_w[1]
                        #setting shape number from no of capsule to number of kernels
                        tmp_reshp_w[1]=tmp_reshp_w[0]
                        
                        cap_net_layer_info[tmp_lyr_nm]=cap_array.reshape(tuple(tmp_reshp_w.astype(np.int)))
                    else:
                        cap_net_layer_info[tmp_lyr_nm]=cap_array.reshape(tuple(tmp_lyr_shp.astype(np.int)))
                except ValueError as e:
                    ipdb.set_trace()
                #Deleting old capsule layer with linear shape replacing with convoled layer with correct name
                del cap_net_layer_info[cap_layer]
                del summ_dict[epch_no][cap_layer]
                #Determine if bias channel or weight channel being assessed
                               #Getting summary info  based on channel information
                stat_dict=per_caps_stat(cap_net_layer_info[tmp_lyr_nm],
                                        chnl_det,no_capsules)
                #Assigning summary information to file
                
                if tmp_lyr_nm not in summ_dict[epch_no].keys():
                    summ_dict[epch_no][tmp_lyr_nm]=stat_dict
                else:
                    i+=1
                    tmp_lyr_nm_pl=tmp_lyr_nm+str(i)
                    summ_dict[epch_no][tmp_lyr_nm_pl]=stat_dict
                
            #getting summary info based on newly reshaped system. 
            else:
                print('layer not present in dictionary')
                
    return tmp_epoch_dict,summ_dict
    
def write_to_output(exp_nm,parse_args,**kwargs):
    """The purpose of this method is to write the main summary dictionary to file based on the output directory argument"""
    
    #ipdb.set_trace()
    for f_nms,val in kwargs.items():
        tmp_path=os.path.join(parse_args.output_dir,exp_nm+'_'+f_nms)
        
        with open(tmp_path,'wb') as fb:
            pickle.dump(val,fb)
    
    
def det_bias_wght_lyr(cap_layer,args,
                      sub_str_dict={'capsule_bias_channel':['bconv','bdeconv','bprimary'],
                                    'capsule_wght_channel':['wconv','wdeconv','wprimary']}):
    """The purpose of this method is to determine if a layer is bias or weight layer based on its name"""            
    #ipdb.set_trace()
    args_dict=vars(args)
    
    for k,sub_str_lst in sub_str_dict.items():
        tmp_lst=[x for x in sub_str_lst if cap_layer.lower().find(x)!=-1]
        
        if len(tmp_lst)>0:
            ret_val=args_dict[k]
            break
        
    return k,ret_val

    
def per_caps_stat(tmp_arr,chnl_det,no_caps):
    
    index_slc=tmp_arr.shape[chnl_det]
    
    tmp_dict_stat={}
    if chnl_det==3:
        #ipdb.set_trace()
        slc_iter=int(index_slc/no_caps)
        for i in range(0,
                        index_slc,
                        slc_iter):
            tmp_dict_stat['capsule_no_'+str(i)]=gen_per_cap_stat(tmp_arr[:,:,:,i:i+slc_iter].flatten())
            
            
    elif chnl_det==2:
        
        for i in range(0,index_slc):    
            
            tmp_dict_stat['capsule_no_'+str(i)]=gen_per_cap_stat(tmp_arr[:,:,i,:].flatten())
            
    return tmp_dict_stat
        
        
            
        
def get_layr_shp(cap_layer,w_b_dict):
    #finds capsule layer via string matching of subtring filenames. 
    
    return [(k,w_b_dict[k]) for k,v in w_b_dict.items() if cap_layer.lower().find(k.lower())!=-1]
    
    
    
       
    

def gen_wght_bias_dict(args):
    """The purpose of this method is to generate a weights bias for evaluation against generated bias and weight""" 
    
    
    with open(os.path.join(args.wght_bias_dir,
                           'segcaps_weights_shape_orig.pickle'),'rb') as fb:
        w_dict=pickle.load(fb)
        
    with open(os.path.join(args.wght_bias_dir,
                           'segcaps_bias_shape_orig.pickle'),'rb') as fb:
        b_dict=pickle.load(fb)
        
    w_dict.update(b_dict)
    
    return w_dict
           
            
def eval_weights_dict(tmp_epoch_dict:dict)->dict:
    """The purpose of this function is to evaluate and pull different weights from an interactive tf session into numby arrays and to then output these arrays to final pickle file"""
    #ipdb.set_trace()
    sess = tf.InteractiveSession()
    
    for k,v in tmp_epoch_dict.items():
        
        reader = pywrap_tensorflow.NewCheckpointReader(v)
        var_to_shape_map = reader.get_variable_to_shape_map()
        tmp_dict={}
        #Creating nested dictionary between weights and bias to be evaluated
        for key in var_to_shape_map:
            tmp_dict[key]=reader.get_tensor(key)
        
        tmp_epoch_dict[k]=tmp_dict
        
    sess.close()
    
    return tmp_epoch_dict
            
            
        
def get_epoch_dict_raw(file_lst:list,args,key_word='model-')->list:
    """The purpose of this method is to take a file list and find the
    total number of epochs present in the weights list once this is taken they
    can then be utilised for writing the numpy values present in the images to file."""
    #ipdb.set_trace()
    #Getting model number only to directory. unique numbers only for each model number
    tmp_lst=set([os.path.join(os.path.dirname(x),
                          os.path.splitext(os.path.basename(x))[0]) for x in file_lst])
    #Getting the index of where the keyword is present in the file 
    tmp_lst_idx=[x[x.find(key_word)+len(key_word):]for x in tmp_lst]
    #integer version of list
    final_epoch_list=[int(x)/args.sample_per_epoch for x in tmp_lst_idx]
    
    return dict(zip(final_epoch_list,tmp_lst))
    

#Argparse main starting point. 
if __name__=="__main__":
    
    arg_vals=parse_arg()
    try:
 
        main(arg_vals)
    except ValueError as e:
        print('error',e)
        ipdb.set_trace()
    
    
    
