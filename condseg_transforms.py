# Displayed and Rand Transform, decorator @patch
from fastai.basics import *
from fastai import *


from transforms import AddChannel, Iso, PadSz,\
                       ZScale, \
                       GNoise, GBlur,\
                       RandBright, RandContrast, \
                       RandDihedral, MattAff
# numpy and torch
import numpy as np
import torch

class AddAtlas(ItemTransform):
    split_idx = None
    
    def __init__(self, items):
        self.n = len(items)
        self.items = items
    
    def encodes(self, item):
        atlas_item = self.items[np.random.randint(self.n)]
        return tuple(item), tuple(atlas_item)
    
class IsoAtlas(ItemTransform):
    split_idx = None
    
    def __init__(self, new_sp = 2):
        self.new_sp = new_sp
        
    def encodes(self, x):
        input1, atlas1 = x
        return Iso(self.new_sp)(input1), Iso(self.new_sp)(atlas1)

class ZScaleAtlas(ItemTransform):
    split_idx = None
    
    """
    normalize a target image by subtracting the mean of foreground pixels
    and dividing by the standard deviation
    Source: https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/zscore.py
    """
 
    # batch BCDHW
    def encodes(self, x):
        input1, atlas1 = x    
        return ZScale()(input1), ZScale()(atlas1)
    
class AddChAtlas(ItemTransform):
    split_idx = None
    
    def encodes(self, x):
        input1, atlas1 = x
        return AddChannel()(input1), AddChannel()(atlas1)
    
class MattAffAtlas(ItemTransform):
    split_idx = 0
    
    def __init__(self, p=0.5, strength=0.05):
        self.p        = p
        self.strength = strength
    
    def encodes(self, x):
        input1, atlas1 = x
        return MattAff(p=self.p, strength=self.strength)(input1), atlas1

class ConcatChannel(ItemTransform):
    split_idx = None
    
    def encodes(self, x):
        input1, atlas1 = x
        mr,mk = input1
        atlas_mr, atlas_mk = atlas1
        return torch.cat([mr, atlas_mr, atlas_mk], dim=1), mk