import sys
sys.path.append("./Lung_Segmentation")

from collections import OrderedDict
import tensorflow as tf
import numpy as np
import json
import tensorflow_train.utils.tensorflow_util
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.losses.semantic_segmentation_losses import softmax, weighted_softmax, spread_loss, weighted_spread_loss,focal_loss_fixed,generalised_dice_loss
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.summary_handler import SummaryHandler, create_summary_placeholder
from utils.segmentation.segmentation_test import SegmentationTest
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from utils.segmentation.metrics import DiceMetric
import utils.io.image

from LungSeg.dataset import Dataset
from LungSeg.cnn_network import network_ud
from LungSeg.capsule_network import Matwo_CapsNet, MatVec_CapsNet
from LungSeg.SegCaps.SegCaps import SegCaps_multilabels

class MainLoop(MainLoopBase):
    def __init__(self, param):
        super().__init__()
       
        self.loss_function=param['loss_function']#usually param[0]
        self.network=param['network']#Usually param[1]
        self.routing_type=param['routing_type'] #usually param[2]
        #self.output_folder=param['output_folder']
        self.current_iter=0#param['current_iter']
        self.batch_size = param['batch_size'] #abritrary value is 1
        self.learning_rates = param['learning_rates'] #arbitrary value is [1,1]
        self.max_iter = param['max_iter']#arbitrary is 300000
        self.test_iter =param['test_iter']#arbitrary is 10000
        self.cls_wghts=param['class_weights_arr']
        self.disp_iter = 100
        self.data_aug=param['data_aug']
        self.patience=param['patience']#Patience for running of analysis
        self.dice_score_earlystop=param['earlystop_sp']
        
        self.agg_dice_score=None#Additional aggregate dice score added to mitigate against overfitting when data augmentation is introduced. 
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.num_labels = param['num_labels']#6
        self.data_format = param['data_format']#usually 'channels_first' #WARNING: Capsule might not work with channel last ! 
        self.channel_axis = 1
        self.save_debug_images = param['save_debug_images']
        self.base_folder = "./Dataset/"
        self.image_size = param['image_size']#[128, 128] 
        self.image_spacing = [1, 1]
        self.output_folder = './Experiments/' + self.network.__name__ + '_' + self.output_folder_timestamp()
        #Opening augmentation dictionary for analysis
        with open(param['aug_dict_path'],'r') as fb:
            self.aug_dict=json.load(fb)
        
        
        self.dataset = Dataset(image_size = self.image_size,
                               image_spacing = self.image_spacing,
                               num_labels = self.num_labels,
                               base_folder = self.base_folder,
                               data_format = self.data_format,
                               save_debug_images = self.save_debug_images,
                               data_aug=self.data_aug,
                               transform_dict=self.aug_dict)

        self.dataset_train = self.dataset.dataset_train()
        self.dataset_train.get_next()
        self.dataset_val = self.dataset.dataset_val()
        self.dice_names = list(map(lambda x: 'dice_{}'.format(x), range(self.num_labels)))
        self.additional_summaries_placeholders_val = dict([(name, create_summary_placeholder(name)) for name in self.dice_names])

        if self.network.__name__ is 'network_ud' :
            self.net_file = './Lung_Segmentation/LungSeg/cnn_network.py'
        elif self.network.__name__ is 'SegCaps_multilabels' :
            self.net_file = './Lung_Segmentation/LungSeg/SegCaps/SegCaps.py'
        else:
            self.net_file = './Lung_Segmentation/LungSeg/capsule_network.py'
        self.files_to_copy = ['main_train_and_test.py', self.net_file]

    def initNetworks(self):
        network_image_size = list(reversed(self.image_size))
        global_step = tf.Variable(self.current_iter)

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('data', [1] + network_image_size),
                                                  ('mask', [self.num_labels] + network_image_size)])

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build train graph
        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size)
        data, mask = self.train_queue.dequeue()
        with tf.variable_scope('training',reuse=False):
            if self.network.__name__ is 'network_ud' or self.network.__name__ is 'SegCaps_multilabels' :
                    prediction = training_net(data,num_labels=self.num_labels,is_training=True, data_format=self.data_format)
            else:
                    prediction = training_net(data, routing_type=self.routing_type, num_labels=self.num_labels, is_training=True, data_format=self.data_format)

        #Print parameters count
        print ('------------')
        var_num=np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()])
        print ('Net number of parameter : '+ str(var_num))

        # losses
        if 'weighted_spread_loss' in self.loss_function.__name__:
            self.loss_net = self.loss_function(labels=mask, logits=prediction,
                                               global_step=global_step,
                                               data_format=self.data_format,w_l=self.cls_wghts)
        elif 'weighted_softmax' in self.loss_function.__name__:
            self.loss_net = self.loss_function(labels=mask, logits=prediction,
                                               data_format=self.data_format,w_l=self.cls_wghts)
        
        elif 'spread_loss' in self.loss_function.__name__ :
            self.loss_net = self.loss_function(labels=mask, logits=prediction,
                                               global_step=global_step,data_format=self.data_format)
        else:
            self.loss_net = self.loss_function(labels=mask, logits=prediction, data_format=self.data_format)
        print('LOSS FUNCTION UTILISED:',self.loss_function.__name__)
        print('LEARNING RATE VALUES:',self.learning_rates)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
                self.loss = self.loss_net

        self.train_losses = OrderedDict([('loss', self.loss_net)])

        # solver
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rates).minimize(self.loss, global_step=global_step,var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='training'))

        # build val graph
        val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries, shape_prefix=[1])
        self.data_val = val_placeholders['data']

        with tf.variable_scope('testing',reuse=True):

            if self.network.__name__ is 'network_ud' or self.network.__name__ is 'SegCaps_multilabels' :
                self.prediction_val = training_net(self.data_val, num_labels=self.num_labels, is_training=False, data_format=self.data_format)
            else:
                self.prediction_val = training_net(self.data_val, routing_type=self.routing_type, num_labels=self.num_labels, is_training=False, data_format=self.data_format)
            self.mask_val = val_placeholders['mask'] 

            # losses
            self.loss_val = self.loss_function(labels=self.mask_val, logits=self.prediction_val, data_format=self.data_format)
            self.val_losses = OrderedDict([('loss', self.loss_val)])


    def test(self):
        print('Testing...')
        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        labels = list(range(self.num_labels))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator='cubic',
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)
        segmentation_statistics = SegmentationStatistics(labels,
                                                         self.output_folder_for_current_iteration(),
                                                         metrics={'dice': DiceMetric()})
        num_entries = self.dataset_val.num_entries()
        for i in range(num_entries):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            generators = dataset_entry['generators']
            transformations = dataset_entry['transformations']

            feed_dict = {self.data_val: np.expand_dims(generators['data'], axis=0),
            self.mask_val: np.expand_dims(generators['mask'], axis=0)}
                # run loss and update loss accumulators
            run_tuple = self.sess.run((self.prediction_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(),
                                          feed_dict=feed_dict)
           
            prediction = np.squeeze(run_tuple[0], axis=0)
            input = datasources['image']
            transformation = transformations['data']
            prediction_labels, prediction_sitk = segmentation_test.get_label_image(prediction, input, self.image_spacing, transformation, return_transformed_sitk=True)
            utils.io.image.write_np_colormask(prediction_labels, self.output_file_for_current_iteration(current_id + '.png'))
            utils.io.image.write_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'))

            groundtruth = datasources['mask']
            segmentation_statistics.add_labels(current_id, prediction_labels, groundtruth)
            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, num_entries, prefix='Testing ', suffix=' complete')

        # finalize loss values
        segmentation_statistics.finalize()
        dice_list = segmentation_statistics.get_metric_mean_list('dice')
        #Additional logic added to trip dice score based on early stopping criteria to prevent overfitting. 
        if self.agg_dice_score is None:
            self.agg_dice_score=np.array(dice_list)
        else:
            dice_score_trip=sum(np.subtract(self.agg_dice_score,np.array(dice_list)))
            print('Dice score trip value:',dice_score_trip)
            if dice_score_trip<=self.dice_score_earlystop and self.current_iter>=self.patience:
                print('Initiating early stopping dice score tripped:',dice_score_trip)
                self.current_iter=self.max_iter+1
            else:
                self.agg_dice_score=np.mean([self.agg_dice_score,np.array(dice_list)],axis=0)
        
        dice_dict = OrderedDict(list(zip(self.dice_names, dice_list)))
        self.val_loss_aggregator.finalize(self.current_iter, summary_values=dice_dict)

if __name__ == '__main__':
    #parameter=[[softmax,network_ud,'']]
    #parameter=[[spread_loss,Matwo_CapsNet,'dual']]
    #parameter=[[spread_loss,MatVec_CapsNet,'dynamic']]
    #parameter=[[spread_loss,MatVec_CapsNet,'dual']]

    #parameter=[[weighted_spread_loss,SegCaps_multilabels,'']]
    
    grid_search_parameter=[{'loss_function':weighted_softmax,'network':SegCaps_multilabels,'model_file_path':'',
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.5,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([1,31.64997857, 292.64977218, 284.74978171,183.73985934])},
        {'loss_function':weighted_softmax,'network':SegCaps_multilabels,'model_file_path':'',
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.0003,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([1,31.64997857, 292.64977218, 284.74978171,183.73985934])},
     {'loss_function':weighted_softmax,'network':SegCaps_multilabels,'model_file_path':'',
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.9,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([1,31.64997857, 292.64977218, 284.74978171,183.73985934])},
      {'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,'model_file_path':'',
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.5,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([1,31.64997857, 292.64977218, 284.74978171,183.73985934])},
     {'loss_function':weighted_spread_loss,'network':SegCaps_multilabels,'model_file_path':'',
            'routing_type':'','batch_size':1,'max_iter':7500,
            'test_iter':250,'data_aug':True,
            'num_labels':5,'learning_rates':0.9,
            'data_format':'channels_first',
            'save_debug_images':False,'image_size':[256,256],
            'aug_dict_path':'./aug_dict_prob.json','patience':5000,'earlystop_sp':0.001,
            'class_weights_arr':np.array([1,31.64997857, 292.64977218, 284.74978171,183.73985934])}]
    
    
    for param in grid_search_parameter:
            loop = MainLoop(param)
            loop.run()
