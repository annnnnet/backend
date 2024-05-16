from __future__ import division
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class PIL2Numpy(object):
    def __init__(self):
        pass

    def __call__(self, pil_image):
        return np.array(pil_image)

class Resize(object):
    def __init__(self, w=224, h=224, interpolation=Image.BILINEAR):
        self.w = w
        self.h = h
        self.interpolation = interpolation
    def __call__(self, image):
        return image.resize((self.w, self.h), self.interpolation)
