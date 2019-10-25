# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:05:27 2019

@author: aczd087
"""

                    'scale':[[True],'rotation':[True,0.35],
                    'scale_uniform':[True,0.2],
                    'Scale':[True,[0.1,0.1]],
                    'Translation':[True],
                    'deformation':[True,[8,8],15]
                    
                    
                            trial_dict={'translation_input_centre':[True,translation.InputCenterToOrigin(self.dim)],
                    'scale_fit':[True,scale.Fit(self.dim, self.image_size,
                                                self.image_spacing)],
                    'translation_random':[True,translation.Random(self.dim,
                                                                  [20, 20])],
                    'rotation_random':[False,rotation.Random(self.dim,
                                                             [0.35])],
                    'scale_random_uniform':[False,scale.RandomUniform(self.dim,
                                                                      0.2)],
                    'scale_random':[False,scale.Random(self.dim,
                                                       [0.1, 0.1])],
                    'translation_origin_center':[False,translation.OriginToOutputCenter(self.dim,
                                                                                        self.image_size,
                                                                                        self.image_spacing)],
                    'deformation':[False,deformation.Output(self.dim,
                                                            [8, 8],
                                                            15,
                                                            self.image_size,
                                                            self.image_spacing)]}