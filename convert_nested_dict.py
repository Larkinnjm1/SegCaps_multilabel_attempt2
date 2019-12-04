# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:02:20 2019

@author: aczd087
"""

import argparse
import pandas as pd 
import os 
import pickle

def parse_args():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-i','--input_file',help='Name of file to be converted from nested dictoinary to regular file',type=str,
                        required=True)
    parser.add_argument('-o','--output_file',help='Write file for analysis',type=str,required=True)
    
    return parser.parse_args()


def Main(args):
    
    dict_fl=read_file_pickle(args)
    
    df_reform=rewrite_df(dict_fl)
    
    write_file_output(df_reform,args)
    
def write_file_output(df_reform,args):
    
    """Write final file converted to file"""
    fl_nm=os.path.splitext(os.path.basename(args.input_file))[0]
    
    df_reform.to_csv(os.path.join(args.output_file,fl_nm+'_df.csv'))
    

def read_file_pickle(args):
        
    with open(args.input_file,'rb') as fb:
        trl_dict=pickle.load(fb)

    return trl_dict


def rewrite_df(trl_dict):
    
    
    final_lst=[]
    for epch_no,epch_data in trl_dict.items():
        if isinstance(epch_data,dict):
            for cap_lyr,cap_data in epch_data.items():
                if isinstance(cap_data,dict):
    
                    for per_cap_no,per_cap_data in cap_data.items():
                        tmp_dict={}
                        tmp_dict['cap_no']=per_cap_no
                        tmp_dict.update(per_cap_data)
                        tmp_dict['layer_name']=cap_lyr
                        tmp_dict['epoch_no']=epch_no
                        final_lst.append(tmp_dict)
                        
    return pd.DataFrame(final_lst)
    
if __name__=='__main__':
    
    args=parse_args()
    
    Main(args)
    