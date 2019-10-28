import os
import sys
import numpy as np
import SimpleITK as sitk
import ipdb
sys.path.append(r"C:\Users\aczd087\Downloads\SegCaps_multilabel_attempt2\Lung_Segmentation")
from datasets.reference_image_transformation_dataset import ReferenceTransformationDataset
from datasources.cached_image_datasource import CachedImageDataSource
from generators.image_generator import ImageGenerator
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.spatial import translation, scale, composite, rotation, deformation,flip
from utils.np_image import split_label_image, distance_transform
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from transformations.intensity.np.normalize import normalize_robust
import random

class Dataset(object):
    """
    The dataset that processes files from the MMWHS challenge.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 num_labels,
                 base_folder,
                 data_aug,
                 transform_dict,
                 input_gaussian_sigma=1.0,
                 label_gaussian_sigma=1.0,
                 data_format='channels_first',
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param image_spacing: Network input image spacing.
        :param base_folder: Dataset base folder.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param label_gaussian_sigma: Sigma value for label smoothing.
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.data_format = data_format
        self.data_aug=data_aug
        self.save_debug_images = save_debug_images
        self.dim = 2
        self.num_labels =num_labels
        self.transform_dict=transform_dict
        self.image_base_folder = os.path.join(self.base_folder,'images')
        self.setup_base_folder = os.path.join(self.base_folder, 'setup')
        self.mask_base_folder = os.path.join(self.base_folder,'masks')
        self.postprocessing_random = self.intensity_postprocessing_mr_random
        self.postprocessing = self.intensity_postprocessing_mr
        self.train_file = os.path.join(self.setup_base_folder,'fold1.txt')
        self.test_file = os.path.join(self.setup_base_folder,'fold2.txt')

    def datasources(self):
        """
        Returns the data sources that load data.
        {
        'image:' CachedImageDataSource that loads the image files.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        'mask:' CachedImageDataSource that loads the groundtruth labels.
        }
        :return: A dict of data sources.
        """
        preprocessing = lambda image: gaussian_sitk(image, self.input_gaussian_sigma)
        image_datasource = CachedImageDataSource(self.image_base_folder, '', '', '.png', preprocessing=preprocessing)
        mask_datasource = CachedImageDataSource(self.mask_base_folder, '', '', '.png', sitk_pixel_type=sitk.sitkUInt8)
        return {'image': image_datasource,
                'mask': mask_datasource}

    def data_generators(self, image_post_processing, mask_post_processing):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :return: A dict of data generators.
        """
        image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, data_format=self.data_format)
        mask_image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='nearest', post_processing_np=mask_post_processing, data_format=self.data_format)
        return {'data': image_generator,
                'mask': mask_image_generator}

    def data_generator_sources(self):
        """
        Returns a dict that defines the connection between datasources and datagenerator parameters for their get() function.
        :return: A dict.
        """
        return {'data': {'image': 'image'},
                'mask': {'image': 'mask'}}

    def split_labels(self, image):
        """
        Splits a groundtruth label image into a stack of one-hot encoded images.
        :param image: The groundtruth label image.
        :return: The one-hot encoded image.
        """
        split = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
        split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
        smoothed = np.stack(split_smoothed, 0)
        image_smoothed = np.argmax(smoothed, axis=0)
        split = split_label_image(image_smoothed, list(range(self.num_labels)), np.uint8)
        return np.stack(split, 0)

    def intensity_postprocessing_mr_random(self, image):
        """
        Intensity postprocessing for MR input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        
        shiftscale_func=self.mod_pst_prc_mr_rand_args()
        
        return shiftscale_func(image)
        
    def mod_pst_prc_mr_rand_args(self):
        
        #random_shift=0.2
        #random_scale=0.4
        #clamp_min=-1.0
        
        return ShiftScaleClamp(self.transform_dict['intensity']['random_shift'],
                               self.transform_dict['intensity']['random_scale'],
                              self.transform_dict['intensity']['clamp_min'])

    def intensity_postprocessing_mr(self, image):
        """
        Intensity postprocessing for MR input.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(clamp_min=-1.0)(image)

    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :return: The transformation.
        """
        
        commands=self.mod_spat_aug()
        print('Commands being utilised for analysis:',commands)
        
        return composite.Composite(self.dim,
                                   commands)
        
    
    def mod_spat_aug(self):
        """
        The purpose of this method is to take an dictoinary input from the user to determine data augmentation that will
        be utilised in the process for analysis
        """
        
        
        trial_dict={'translation_input_centre':[self.transform_dict['spatial']['trans_input_centre_bool'],0.001
                                                translation.InputCenterToOrigin(self.dim)],
                    'scale_fit':[self.transform_dict['spatial']['scale_fit_bool'],
                                 self.transform_dict['spatial']['scale_fit_prob_thresh_sp'],
                                 scale.Fit(self.dim,
                                           self.image_size,
                                           self.image_spacing)],
                    'translation_random':[self.transform_dict['spatial']['trans_rand_bool'],
                                          self.transform_dict['spatial']['trans_rand_fit_prob_thresh_sp'],
                                          translation.Random(self.dim,
                                                             self.transform_dict['spatial']['trans_rand_random_ofs_arg'])],
                    'rotation_random':[self.transform_dict['spatial']['rot_rand_bool'],
                                       self.transform_dict['spatial']['rot_rand_prob_thresh_sp'],
                                       rotation.Random(self.dim,
                                                       self.transform_dict['spatial']['rotation_random_angle_arg'])],
                    'flip_random':[self.transform_dict['spatial']['flip_rand_bool'],
                                   self.transform_dict['spatial']['flip_rand_prob_thresh_sp'],
                                   self.flip.Random(self.dim,
                                                    self.transform_dict['spatial']['flip_random_scaling_parameter'])],
                    'scale_random_uniform':[self.transform_dict['spatial']['rand_scal_uni_bool'],
                                            self.transform_dict['spatial']['rand_scal_prob_thresh'],
                                            scale.RandomUniform(self.dim,
                                                                self.transform_dict['spatial']['random_scale_uniform_arg'])],
                    'translation_origin_center':[self.transform_dict['spatial']['translate_orig_centre_bool'],
                                                 self.transform_dict['spatial']['translate_orig_centre_thresh'],
                                                 translation.OriginToOutputCenter(self.dim,
                                                                                  self.image_size,
                                                                                  self.image_spacing)],
                    'deformation':[self.transform_dict['spatial']['deform_bool'],
                                   self.transform_dict['spatial']['deform_prob_thresh'],
                                   deformation.Output(self.dim,
                                                      self.transform_dict['spatial']['deformation_key_nodes_arg'],
                                                      self.transform_dict['spatial']['deformation_max_deform_arg'],
                                                      self.image_size,
                                                      self.image_spacing)]} #
        
        
        select_trans=[]
        #ipdb.set_trace()
        for k,v in trial_dict.items():
            
            if v[0]==True:
                select_trans.append((v[1],v[2]))
            else:
                continue
        
        return select_trans
    
    

    def spatial_transformation(self):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.Fit(self.dim,self.image_size, self.image_spacing),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing)]
                                   )

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_file, random=True, keys=['image_id'])
        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        #Generating data source information around analysis
        if self.data_aug==False:
            generators=self.data_generators(self.postprocessing,self.split_labels)
            reference_transformation=self.spatial_transformation()
        
        elif self.data_aug==True:
            generators = self.data_generators(self.postprocessing_random, self.split_labels)
            reference_transformation = self.spatial_transformation_augmented()
        

        return ReferenceTransformationDataset(dim=self.dim,
                                              reference_datasource_keys={'image': 'image'},
                                              reference_transformation=reference_transformation,
                                              datasources=sources,
                                              data_generators=generators,
                                              data_generator_sources=generator_sources,
                                              iterator=iterator,
                                              debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'])
        sources = self.datasources()
        generator_sources = self.data_generator_sources()
        generators = self.data_generators(self.postprocessing, self.split_labels)
        reference_transformation = self.spatial_transformation()

        return ReferenceTransformationDataset(dim=self.dim,
                                              reference_datasource_keys={'image': 'image'},
                                              reference_transformation=reference_transformation,
                                              datasources=sources,
                                              data_generators=generators,
                                              data_generator_sources=generator_sources,
                                              iterator=iterator,
                                              debug_image_folder='debug_val' if self.save_debug_images else None)
