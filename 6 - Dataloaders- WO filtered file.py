#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# The goal of this notebook is to train a 3d UNET segmentation model to output binary mask representing the sella turcica ROI.
# 
# Notes:
# - following https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# In[2]:


# !which python


# In[4]:


from platform import python_version

print(python_version())


# In[1]:


# Check GPU stats

from pynvml import *
nvmlInit()
try:
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("Device", i, ":", nvmlDeviceGetName(handle))
except NVMLError as error:
    print(error)
    
# https://docs.fast.ai/dev/gpu.html
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

import torch
print(torch.cuda.is_available() )
torch.cuda.empty_cache()
torch.cuda.set_device(0)


# In[8]:


# Get path to 4 TB HD

# /media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home

# wsl: /home/rgologorsky/DeepPit
hd_path = "../" * 5 + "/media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a" + "/home/" + "rachel/PitMRdata"

# all folders in HD
all_folders = os.listdir(hd_path)

print(all_folders)

# labels
print(os.listdir(f"{hd_path}/samir_labels"))


# In[9]:


from faimed3d.all import *
from fastai import *
from torchvision.models.video import r3d_18
from fastai.callback.all import SaveModelCallback
from torch import nn

import os
import time
import pickle
from pathlib import Path

import SimpleITK as sitk

import numpy as np
import pandas as pd
from pandas import DataFrame as DF

from helpers_preprocess import get_data_dict
from helpers_general import sitk2np, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups

# imports
from helpers_general import sitk2np, np2sitk, round_tuple, lrange, get_roi_range, numbers2groups
from helpers_preprocess import mask2bbox, print_bbox, get_bbox_size, print_bbox_size, get_data_dict, folder2objs
from helpers_viz import viz_axis


# In[10]:


train_path = f"{hd_path}/samir_labels/50155-50212/"
data_dict_full = get_data_dict(train_path)


# In[11]:


data_dict = {k:v for k,v in list(data_dict_full.items())[:6]}


# In[12]:


# n = # (img, mask) data points
n = len(data_dict.keys())
        
# print train data dict
print(f"N = {n}")
print(f"Train data folders: {numbers2groups(sorted([int(x) for x in os.listdir(train_path)]))}")
print(f"Training data: key = train folder, value = full path to (segm obj, nii file)\n")

for folder_name, (obj_path, nii_path) in list(data_dict.items())[:5]:
    print(f"Folder {folder_name}: ", "\n\t", obj_path, "\n\t", nii_path, "\n")


# In[13]:


info_df = pd.read_pickle("./50155-50212.pkl")
print(info_df)


# In[14]:


# Reference frame:  [191, 268, 268] 2 [0. 0. 0.] [1. 1. 1.] [1. 0. 0. 0. 1. 0. 0. 0. 1.] [ 95.5 134.  134. ]
# Reference center:  [ 95.5 134.  134. ]


# In[15]:


d = pd.DataFrame(data_dict.values(), columns = ["masks", "images"])
print(d)


# In[16]:


# 20% validation set, 80% training set
d['is_valid'] = np.random.choice(2, n, p = [0.8,0.2])
print(d)


# In[17]:


# add metadata
def lookup_metadata(img_path):
    sz, sp = info_df.loc[info_df["fn"] == img_path][["sz", "sp"]].values[0]
    return sz, sp


# In[18]:


# which sizes are represented?
szs, spcs = zip(*[lookup_metadata(img_path) for img_path in d["images"].tolist()])
unique_szs  = set(szs)
unique_spcs = set(spcs)
print(f"Sizes ({len(unique_szs)}): ", *list(unique_szs)[:10], sep="\n")
print(f"Spacings ({len(unique_spcs)}): ", *list(unique_spcs)[:10], sep="\n")

unique, idxs, cnts = np.unique(spcs, return_index=True, return_inverse=False, return_counts=True, axis=0)
print("Unique: ", unique)
print("Idxs: ", idxs)
print("Counts: ", cnts)


# In[ ]:





# # Resize to common dimension

# In[19]:


# img data = size and spacing
all_img_data = [lookup_metadata(img_path) for img_path in d["images"].tolist()]


# In[20]:


def get_reference_frame(all_img_data):
    img_data = all_img_data
    
    dimension = 3 # 3D MRs
    pixel_id = 2 # 16-bit signed integer

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    for img_sz, img_spc in img_data:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx else mx                                       for sz, spc, mx in zip(img_sz, img_spc, reference_physical_size)]
    
    # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Isotropic (1,1,1) pixels
    reference_spacing = np.ones(dimension)
    reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    # Set reference image attributes
    reference_image = sitk.Image(reference_size, pixel_id)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    return reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center

def get_reference_image(reference_frame):
    reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center = reference_frame
    reference_image = sitk.Image(reference_size, pixel_id)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    return reference_image, reference_center

# Get reference frame
reference_frame = get_reference_frame(all_img_data)
print("Reference frame: ", *reference_frame)

# Get reference image
reference_image, reference_center = get_reference_image(reference_frame)
print("Reference center: ", reference_center)

# Print info on reference image
print_sitk_info(reference_image)


# In[21]:


# removed mask arg
def resample2ref(img, reference_image, reference_center, interpolator = sitk.sitkLinear, default_intensity_value = 0.0, dimension=3):
    
    # Define translation transform mapping origins from reference_image to the current img
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_image.GetOrigin())
    
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    
    return sitk.Resample(img, reference_image, centered_transform, interpolator, default_intensity_value, img.GetPixelID())

# Composite tfm - https://simpleitk.readthedocs.io/en/master/migrationGuide2.0.html
def resample2ref_item(img, mask, reference_image, reference_center, interpolator = sitk.sitkLinear, default_intensity_value = 0.0, dimension=3):
    
    # Define translation transform mapping origins from reference_image to the current img
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_image.GetOrigin())
    
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    
    # sitk 1.x
    #centered_transform.AddTransform(centering_transform)
    
    # sitk 2.x
    centered_transform = sitk.CompositeTransform([centered_transform, centering_transform])
    
    objs    = img, mask
    interps = interpolator, sitk.sitkNearestNeighbor
    return [sitk.Resample(o, reference_image, centered_transform, interp, default_intensity_value, o.GetPixelID()) for o,interp in zip(objs, interps)]


# In[22]:


from fastai.vision.all import *


# In[23]:


from helpers_preprocess import folder2objs

class SlicerSegmentationTransform(ItemTransform):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        
    def encodes(self, x):
        obj_path, img_path = x
        folder = Path(obj_path).parent.name
        ras_adj = int(folder) in range(50455, 50464)
        img_obj, mask_obj = folder2objs(folder, self.data_dict, ras_adj)
        return img_obj, mask_obj
    
# class Sitk2Tensir(Transform):
#     def encodes(self, x): return TensorDicom3D.


# In[24]:


# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html
def make_isotropic(image, new_spacing = 1, interpolator = sitk.sitkLinear):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats 
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    new_size = [int(round(osz*ospc/new_spacing)) for osz,ospc in zip(original_size, original_spacing)]
    new_spacing = [new_spacing]*image.GetDimension()
    
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())


# In[25]:


# # test make isotropic: using F.interpolate vs sitk

# file = d["images"][0]

# a = TensorDicom3D.create(file)
# b = a.size_correction(new_spacing=1)
# # p =  PreprocessDicom(correct_spacing=True, spacing = 1)
# # b = p(a)
# print(type(a), a.shape)
# print(type(b), b.shape)

# a1 = sitk.ReadImage(file, sitk.sitkFloat32)
# b1 = make_isotropic(a1)

# print(a1.GetSize(), b1.GetSize())


# In[26]:


# # test make isotropic: using F.interpolate vs sitk

# file = d["images"][1]

# a = TensorDicom3D.create(file)
# b = a.size_correction(new_spacing=1)
# # p =  PreprocessDicom(correct_spacing=True, spacing = 1)
# # b = p(a)
# print(type(a), a.shape)
# print(type(b), b.shape)

# a1 = sitk.ReadImage(file, sitk.sitkFloat32)
# b1 = make_isotropic(a1)

# print(a1.GetSize(), b1.GetSize())


# In[27]:


from helpers_preprocess import folder2objs

class SlicerSegmentationTransform(ItemTransform):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        
    def encodes(self, x):
        obj_path, img_path = x
        folder = Path(obj_path).parent.name
        ras_adj = int(folder) in range(50455, 50464)
        return folder2objs(folder, self.data_dict, ras_adj)
        #img_obj, mask_obj = folder2objs(folder, self.data_dict, ras_adj)
        #return TensorDicom3D(img_obj), TensorDicom3D(mask_obj) 
    
class IsotropicTransform(ItemTransform):
    def encodes(self, x):
        img, mask = x
        return  make_isotropic(img, new_spacing = 1, interpolator = sitk.sitkLinear),                 make_isotropic(img, new_spacing = 1, interpolator = sitk.sitkNearestNeighbor)

class Resample2Ref(ItemTransform):
    def __init__(self, reference_image, reference_center):
        self.ref_im   = reference_image
        self.ref_cntr = reference_center
        
    def encodes(self, x):
        im, mask = x
        return resample2ref_item(im, mask, self.ref_im, self.ref_cntr)
    
class ToTensor3D(ItemTransform):
    def encodes(self, x):
        im, mask = x
        return torch.tensor(sitk.GetArrayFromImage(im)), torch.tensor(sitk.GetArrayFromImage(mask))


# In[28]:


items = list(data_dict.values())
print("Items: ", *items[:5], sep="\n")


# In[29]:


splits = RandomSplitter(seed=42)(items)
print(len(splits[0]), len(splits[1]), splits)


# In[30]:


print("is cuda available?", torch.cuda.is_available() )


# In[31]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[26]:


# tls = TfmdLists(items, [SlicerSegmentationTransform(data_dict), IsotropicTransform, ToTensor3D], splits=splits)


# In[27]:


# Cuda()


# In[32]:


tls = TfmdLists(items, [SlicerSegmentationTransform(data_dict),                         Resample2Ref(reference_image, reference_center),                         ToTensor3D], 
               splits=splits)


# In[33]:


dls = tls.dataloaders(bs=2)


# In[32]:


# dls = dls.cuda()


# In[34]:


b = dls.one_batch()


# In[35]:


print(type(b), b[0].shape, b[1].shape)


# In[36]:


def dice(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection) /
           (iflat.sum() + tflat.sum()))

def dice_score(input, target):
    return dice(input.argmax(1), target)

def dice_loss(input, target): 
    return 1 - dice(input.softmax(1)[:, 1], target)

def loss(input, target):
    return dice_loss(input, target) + nn.CrossEntropyLoss()(input, target[:, 0])


# In[ ]:


learn = unet_learner_3d(dls, r3d_18, n_out=2, 
                        loss_func = loss,
                        metrics = dice_score,
                        model_dir = ".",
                        cbs = [SaveModelCallback(monitor='dice_score')]
                       )
#learn.model.cuda()
#learn = learn.to_fp16()


# In[ ]:


# https://forums.fast.ai/t/very-slow-inference-using-get-preds-in-fastai2/70841/4
# learn.model = learn.model.cuda() will make it GPU enabled.
# learn.dls.to(‘cuda’)
# learn.dls.cuda()


# In[6]:


print("CUDA enabled")


# In[ ]:


learn.model = learn.model.cuda() 
learn.dls.cuda()


# In[ ]:


print("POST CUDA enabled")


# In[ ]:


# learn = unet_learner_3d(dls, r3d_18, n_out=2, 
#                         loss_func = loss,
#                         metrics = dice_score,
#                         model_dir = ".",
#                         cbs = [SaveModelCallback(monitor='dice_score')]
#                        )
# #learn.model.cuda()
# learn = learn.to_fp16()


# In[ ]:


# learn.model


# In[ ]:


# learn.model.cuda()


# In[ ]:


# learn.lr_find()


# In[ ]:


print("PRE learn.fit one cycle")
learn.fit_one_cycle(3, 0.01, wd = 1e-4)


# In[ ]:


print("unfreeze, learn 50")
learn.unfreeze()
learn.fit_one_cycle(50, 1e-3, wd = 1e-4)


# In[ ]:





# In[ ]:


# after_item  = [TensorDicom3D, MaxScale(), resample]
# after_batch = []


# In[ ]:


# # df cols = masks, images, is_valid
# splitter = ColSplitter(2)

# # item tfms
# rescale_method = MaxScale()
# resample = Resample3D(size=(191, 268, 268),
#                       spacing=(1,1,1),
#                       origin=(0,0,0),
#                       direction=(1,0,0,\
#                                  0,1,0,\
#                                  0,0,1))

# items = list(train_data_dict.values())
# tls = TfmdLists(items, [SlicerSegmentationTransform(train_data_dict), IsotropicTransform])

# # batch_tfms=[PreprocessDicom(**kwargs)])
# # item_tfms=AddMaskCodes(codes=codes))


# In[ ]:


# dls = tls.dataloaders(bs=2)


# In[ ]:


# # batch tfms
# batch_tfms = [AddChannel(), \
#               RandomPerspective3D(input_size=268, p=0.5, distortion_scale=0.25), 
#               *aug_transforms_3d(p_all=0.15, noise=False)]

# # dblock
# dblock = DataBlock(blocks=(ImageBlock3D(cls=TensorDicom3D),MaskBlock3D(codes=None)),
#                            get_x=ColReader(0),
#                            get_y=ColReader(1),
#                            splitter=splitter,
#                            item_tfms=item_tfms,
#                            batch_tfms=batch_tfms,
#                            n_inp = 1)


# In[ ]:


# # https://github.com/kbressem/faimed3d/blob/deada354a1ead1341f1578f84ab6325c50be56ca/faimed3d/augment.py
# dls = SegmentationDataLoaders3D.from_df(d, path = '.',
#                                 item_tfms =Resample3D(size=(191, 268, 268),
#                                                       spacing=(1,1,1),
#                                                       origin=(0,0,0),
#                                                       direction=(1,0,0,\
#                                                                  0,1,0,\
#                                                                  0,0,1)),
#                                 batch_tfms = [RandomPerspective3D(input_size=268, p=0.5, distortion_scale=0.25), 
#                                               *aug_transforms_3d(p_all=0.15, noise=False)],
#                                 bs = 1, 
#                                 val_bs = 1,
#                                 splitter = ColSplitter('is_valid'))

