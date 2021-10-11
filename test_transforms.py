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
import SimpleITK as sitk

from pathlib import Path

import SimpleITK as sitk
import meshio

# Displayed and Rand Transform, decorator @patch
from fastai.basics import *
from fastai import *

from fastai.vision.augment import RandTransform

import torchvision as tv
import torchvision.transforms.functional as _F


class IsoTest(Transform):
    split_idx = None
    
    def __init__(self, new_sp = 3):
        self.new_sp = new_sp
        
    def encodes(self, x:str):
        # get sitk objs
        im_path = x
        mr = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im = torch.transpose(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
       
        # resize so isotropic spacing
        orig_sp = mr.GetSpacing()
        orig_sz = mr.GetSize()
        new_sz = [int(round(osz*ospc/self.new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

        while im.ndim < 5: 
            im = im.unsqueeze(0)

        return F.interpolate(im, size = new_sz, mode = 'trilinear', align_corners=False).squeeze()
    
class PadSzTest(ItemTransform):
    split_idx = None
    
    def __init__(self, new_sz):
        self.new_sz = new_sz
    
    def encodes(self, item):
        arr, _ = item
        pad = [x-y for x,y in zip(self.new_sz, arr.shape)]
        pad = [a for amt in pad for a in (amt//2, amt-amt//2)]
        pad.reverse()
        
        res = F.pad(arr, pad, mode='constant', value=0)
        
        return res, res
    
# CONDITIONAL SEGMENTATION IN LIEU OF IMAGE REGISTRATION
class IsoAtlasTest(ItemTransform):
    split_idx = None
    
    def __init__(self, new_sp = 2):
        self.new_sp = new_sp
        
    def encodes(self, x):
        input1, atlas1 = x
        return IsoTest(self.new_sp)(input1), Iso(self.new_sp)(atlas1)

class PadSzAtlasTest(ItemTransform):
    split_idx = None
    
    def __init__(self, maxs):
        self.maxs = maxs
        
    def encodes(self, x):
        input1, atlas1 = x
        return PadSzTest(maxs)(input1), PadSz(maxs)(atlas1)