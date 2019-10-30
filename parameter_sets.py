# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:42:50 2019

@author: aczd087
"""
trian_further_parameter=[{'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,'output_folder':r'XX',
                'routing_type':'','batch_size':1,'max_iter':15000,'current_iter':7500,
                'test_iter':250,'data_aug':True, #250
                'num_labels':5,'learning_rates':0.0003,
                'data_format':'channels_first',
                'save_debug_images':False,'image_size':[256,256], #5000,0.001
                'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
                'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
{'loss_function':weighted_softmax,'network':SegCaps_multilabels,
            'routing_type':'','batch_size':1,'max_iter':15000,'current_iter':7500,'output_folder':r'XX',
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.0003,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])}]








grid_search_parameter=[{'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,'model_file_path':,
                'routing_type':'','batch_size':1,'max_iter':7500,
                'test_iter':250,'data_aug':True, #250
                'num_labels':5,'learning_rates':0.0003,
                'data_format':'channels_first',
                'save_debug_images':False,'image_size':[256,256], #5000,0.001
                'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
                'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
{'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.001,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
 {'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,
        'routing_type':'','batch_size':1,'max_iter':7500,
        'test_iter':250,'data_aug':True,
        'num_labels':5,'learning_rates':0.01,
        'data_format':'channels_first',
        'save_debug_images':False,'image_size':[256,256],
        'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
        'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
 {'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,
        'routing_type':'','batch_size':1,'max_iter':7500,
        'test_iter':250,'data_aug':True,
        'num_labels':5,'learning_rates':0.1,
        'data_format':'channels_first',
        'save_debug_images':False,'image_size':[256,256],
        'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
        'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
{'loss_function':weighted_softmax,'network':SegCaps_multilabels,
                'routing_type':'','batch_size':1,'max_iter':7500,
                'test_iter':250,'data_aug':True,
                'num_labels':5,'learning_rates':0.0003,
                'data_format':'channels_first',
                'save_debug_images':False,'image_size':[256,256],
                'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
                'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
{'loss_function':weighted_softmax,'network':SegCaps_multilabels,
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.001,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
 {'loss_function':weighted_softmax,'network':SegCaps_multilabels,
        'routing_type':'','batch_size':1,'max_iter':7500,
        'test_iter':250,'data_aug':True,
        'num_labels':5,'learning_rates':0.01,
        'data_format':'channels_first',
        'save_debug_images':False,'image_size':[256,256],
        'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
        'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])},
 {'loss_function':weighted_softmax,'network':SegCaps_multilabels,
        'routing_type':'','batch_size':1,'max_iter':7500,
        'test_iter':250,'data_aug':True,
        'num_labels':5,'learning_rates':0.1,
        'data_format':'channels_first',
        'save_debug_images':False,'image_size':[256,256],
        'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
        'class_weights_arr':np.array([0.03987201, 0.36867433, 0.35872208, 0.2314718 , 0.00125978])}]