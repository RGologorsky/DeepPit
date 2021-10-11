# TEST: No input label exists

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

# AddAtlas returns: (mr, None); (atlas_mr, atlas_seg)

class IsoAtlasTest(ItemTransform):
    split_idx = None
    
    def __init__(self, new_sp = 2):
        self.new_sp = new_sp
        
    def encodes(self, x):
        input1, atlas1 = x
        return IsoTest(self.new_sp)(input1), Iso(self.new_sp)(atlas1)
    
class ConcatChannelTest(ItemTransform):
    split_idx = None
    
    def encodes(self, x):
        input1, atlas1 = x
        mr,mk = input1
        atlas_mr, atlas_mk = atlas1
        return torch.cat([mr, atlas_mr, atlas_mk], dim=1), mk