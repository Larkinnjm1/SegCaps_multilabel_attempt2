import SimpleITK as sitk
from transformations.spatial.base import SpatialTransformBase
import random
import ipdb

class Composite(SpatialTransformBase):
    """
    A composite transformation consisting of multiple other consecutive transformations.
    """
    def __init__(self, dim, transformations):
        """
        Initializer.
        :param dim: The dimension of the transform.
        :param transformations: List of other transformations.
        """
        super(Composite, self).__init__(dim)
        self.transformations = transformations

    def get(self, **kwargs):
        """
        Returns the composite sitk transform.
        :param kwargs: Optional parameters sent to the other transformations.
        :return: The composite sitk transform.
        """
        print('Composite function being called for analysis')
        compos = sitk.Transform(self.dim, sitk.sitkIdentity)
        for i in range(len(self.transformations)):
            
            rand_int_sp=random.sample(0,1)
            
            if rand_int_sp>self.transformations[i][0]:
                print('transformation added',self.transformations[i][1])
                compos.AddTransform(self.transformations[i][1].get(**kwargs))
        
        return compos
