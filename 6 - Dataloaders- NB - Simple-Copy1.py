#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# The goal of this notebook is to train a 3d UNET segmentation model to output binary mask representing the sella turcica ROI.
# 
# Notes:
# - following https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# # Prelim
# 
# Check python version

# In[4]:


# get python version
from platform import python_version
print(python_version())


# Check whether GPU enabled.

# In[145]:


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
print("is cuda available?", torch.cuda.is_available() )
torch.cuda.empty_cache()
torch.cuda.set_device(0)

# hm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# # Data Path

# Set path to where data is stored.

# In[6]:


# Get path to 4 TB HD

# /media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home

# wsl: /home/rgologorsky/DeepPit
hd_path = "../" * 5 + "/media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a" + "/home/rachel/PitMRdata"

# all folders in HD
all_folders = os.listdir(hd_path)

print(all_folders)

# labels
print(os.listdir(f"{hd_path}/samir_labels"))


# # Imports

# In[23]:


# imports

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

from helpers_preprocess import get_data_dict, paths2objs, folder2objs

from helpers_general import sitk2np, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups

# imports
from helpers_general import sitk2np, np2sitk, round_tuple, lrange, get_roi_range, numbers2groups
from helpers_preprocess import mask2bbox, print_bbox, get_bbox_size, print_bbox_size, get_data_dict, folder2objs
from helpers_viz import viz_axis


# # Get Items
# 
# Item = (path to MR, path to Segmentation obj)

# In[171]:


# get items = (mask_fn, nii_fn)
train_path = f"{hd_path}/samir_labels/50155-50212/"
data_dict_full = get_data_dict(train_path)

# subset 5 items to test
n_items   = 10
data_dict = {k:v for k,v in list(data_dict_full.items())[:n_items]}
items     = list(data_dict.values())


# ## Metadata
# 
# Check size, spacing, etc of the items in the dataset.

# In[172]:


# load metadata about fns -- size, spacing, etc
full_metadata_df = pd.read_pickle("./50155-50212.pkl")

# filter to items in our dataset
fns = [mr for mr, mk in items]
metadata_df = full_metadata_df[full_metadata_df.fn.isin(fns)]

#metadata_df


# What are the different sizes and spacings in the dataset?

# In[173]:


# which sizes are represented?
szs = np.array(list(metadata_df.sz.values))
sps = np.array(list(metadata_df.sp.values))

def print_unique(vals, sep="*" * 30):
    unique, idxs, cnts = np.unique(vals, return_index=True, return_inverse=False, return_counts=True, axis=0)
    print(sep)
    print("Num unique = ", len(unique))
    print("Unique: ", *unique, sep=" ")
    print("Counts: ", *cnts, sep = " ")
    print("Idxs: ", idxs, sep = " ")
    print(sep)
    
print("Sizes:"); print_unique(szs)
print("Spacings:"); print_unique(sps)


# ## Mask Bbox
# 
# Goal: find the range of slice indices that captures the sella of all inputs. Within a single standardized dataset, the slice indices should be similar.
# 
# Why? A full MR volume has >100 slices. The sella occupies about 10 of them. For quick testing, want the input to be MR with 20 slices and not >100 slices.

# In[174]:


# # list of all bboxs
# ims, mks = zip(*[PathToSitk()(i) for i in items])
# bboxs = [mask2bbox(sitk2np(mk)) for mk in mks]
# descr = ["imin", "imax", "jmin", "jmax", "kmin", "kmax"]

# def bb_dict(bbox): return dict(zip(descr, bbox))

# # dataframe of imin/imax
# mk_df = DF([{'fn': mr, **bb_dict(bboxs[i])} \
#             for i,(mr, mk) in enumerate(items)])

# # irange is 62-108; jrange = 118-181; krange = 60-144
# # => islices 50-150, jslices 100-200, kslices 0-200
# for col in ("i", "j", "k"):
#     col_min = mk_df[f"{col}min"].min()
#     col_max = mk_df[f"{col}max"].max()
#     print(f"{col} ROI Range: {col_min} - {col_max}")

# # plot slice range
# #mk_df.boxplot(column=descr)


# # Transforms
# 
# - PathToSITK (*convert paths to SITK obj*)
# - Resize (*common size, isotropic spacing*)
# - ToTensor (*convert to Pytorch tensor*)
# - TensorSlice & Center Crop (*slice 3d tensor to center part containing sella*)
# - Normalize (*scale image intensities? - diff tissues diff intensities?*)

# In[175]:


# convert mask, img path to SITK objs
class PathToSitk(ItemTransform):
    def encodes(self, x):
        im_path, seg_path = x
        folder = Path(seg_path).parent.name
        ras_adj = int(folder) in range(50455, 50464)
        return paths2objs(im_path, seg_path, ras_adj)

# convert SITK to Tensor3D
class ToTensor3D(Transform):
    def encodes(self, sitk_obj):
        return torch.swapaxes(torch.tensor(sitk.GetArrayFromImage(sitk_obj)), 0, 2)


# In[176]:


# # test PathToSitk
# im_obj, mk_obj = PathToSitk()(items[0])

# print(type(im_obj), type(mk_obj)); print()
# print_sitk_info(im_obj); print()
# print_sitk_info(mk_obj); print()

# # test ToTensor3D
# im = ToTensor3D()(im_obj)

# print(type(im), im.shape)


# In[177]:


# simple isotropic (no common resize)
class IsotropicTfm(ItemTransform):
    
    def __init__(self, new_spacing, interpolator = sitk.sitkLinear): 
        self.spacing      = new_spacing
        self.interpolator = interpolator
    
    def encodes(self, x):
        im, mk = x
        return  im.make_isotropic(self.spacing, self.interpolator),                 mk.make_isotropic(self.spacing, sitk.sitkNearestNeighbor)
    
@patch
def make_isotropic(im:sitk.Image, new_spacing = 1, interpolator = sitk.sitkLinear):
    orig_sz = im.GetSize()
    orig_sp = im.GetSpacing()

    new_sz = [int(round(osz*ospc/new_spacing)) for osz,ospc in zip(orig_sz, orig_sp)]
    new_sp = [new_spacing]*im.GetDimension()
    
    return sitk.Resample(im, new_sz, sitk.Transform(), interpolator,
                         im.GetOrigin(), new_sp, im.GetDirection(), 0.0,
                         im.GetPixelID())


# In[178]:


# # test
# tls = TfmdLists(items[0:2], \
#                 [PathToSitk(), \
#                  IsotropicTfm(1)]
#                )

# im1, mask1 = tls[0]
# im2, mask2 = tls[1]

# # print
# print_sitk_info(im1); print()
# print_sitk_info(mask1); print()

# print_sitk_info(im2); print()
# print_sitk_info(mask2); print()


# # Crop

# In[179]:


# crop center
class CenterCropTfm(Transform):
    def __init__(self, size):
        self.size = size
        
    def encodes(self, arr):
        return self.cropND(arr, self.size)
    
    # https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    @staticmethod
    def cropND(img, bounding):
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]
    
# crop by coords
class CropBBox(Transform):
    def __init__(self, bbox):
        self.bbox = bbox
    
    def encodes(self, arr):
        imin, imax, jmin, jmax, kmin, kmax = self.bbox
        return arr[imin:imax, jmin:jmax, kmin:kmax]
        


# In[180]:


# # test
# tls = TfmdLists(items[0:2], \
#                 [PathToSitk(), \
#                  IsotropicTfm(1), \
#                  ToTensor3D(), \
#                  CenterCropTfm((150, 150, 200))
#                 ])

# im1, mask1 = tls[0]
# im2, mask2 = tls[1]

# # print
# print(type(im1), im1.shape, type(mask1), mask1.shape)
# print(type(im2), im2.shape, type(mask2), mask2.shape)


# ## Mask Bbox
# 
# Goal: find the bbox that captures the general region of the sella. The bbox should be similar within a single standardized dataset.
# 
# Why? A full MR volume has >100 slices. The sella occupies about 10 of them. For quick testing, want the input to be MR with 20 slices and not >100 slices.

# In[181]:


# tls = TfmdLists(items, \
#                 [PathToSitk(), \
#                  IsotropicTfm(1), \
#                  ToTensor3D(), \
#                 ])

# # list of all bboxs
# ims, mks = list(zip(*tls))
# bboxs = [mask2bbox(np.array(mk)) for mk in mks]
# descr = ["imin", "imax", "jmin", "jmax", "kmin", "kmax"]

# def bb_dict(bbox): return dict(zip(descr, bbox))

# # dataframe of imin/imax
# mk_df = DF([{'fn': mr, **bb_dict(bboxs[i])} \
#             for i,(mr, mk) in enumerate(items)])

# # irange is 66-108; jrange = 118-181; krange = 60-144
# # => islices 50-150, jslices 100-200, kslices 0-200
# for col in ("i", "j", "k"):
#     col_min = mk_df[f"{col}min"].min()
#     col_max = mk_df[f"{col}max"].max()
#     print(f"{col} ROI Range: {col_min} - {col_max}")

# # get general bbox of the sella region
# general_bbox = [x for col in ("i", "j", "k") for x in (mk_df[f"{col}min"].min(), mk_df[f"{col}max"].max())]
# print(general_bbox); print_bbox(*general_bbox)

# # plot slice range
# #mk_df.boxplot(column=descr)


# In[182]:


# # test crop
# tls = TfmdLists(items[0:2], \
#                 [PathToSitk(), \
#                  IsotropicTfm(1), \
#                  ToTensor3D(), \
#                  CropBBox(general_bbox)
#                 ])

# ims, mks = list(zip(*tls))

# # print
# for im,mk in zip(ims, mks):
#     print(im.shape, mk.shape)
#     print_bbox(*mask2bbox(np.array(mk)))
#     print()


# In[183]:


# # Viz
# idx = 0
# im_np, mk_np = [np.array(x) for x in (ims[idx], mks[idx])]

# viz_axis(np_arr = im_np, \
#         bin_mask_arr  = mk_np, color1 = "yellow", alpha1=0.3, \
#         slices=lrange(0, mk_np.shape[0]), fixed_axis=0, \
#         axis_fn = np.rot90, \
#         title   = "Axis 0", \
         
#         np_arr_b = im_np, \
#         bin_mask_arr_b  = mk_np,
#         slices_b = lrange(0, mk_np.shape[1]), fixed_axis_b=1, \
#         title_b  = "Axis 1", \
         
#         np_arr_c = im_np, \
#         bin_mask_arr_c  = mk_np,
#         slices_c = lrange(0, mk_np.shape[2]), fixed_axis_c=2, \
#         title_c = "Axis 2", \
         
#         ncols = 5, hspace=0.3, fig_mult=2)


# # Dataloaders
# 
# TODO augmentations.
# 
# - dset = tfms applied to items
# - splits into training/valid
# - bs

# In[243]:


# # imin imax jmin jmax kmin kmax
# print(general_bbox)
# print_bbox(*general_bbox)


# In[262]:


# np.array((66, 118, 60)) + np.array((42, 93, 93))

# np.array((66, 118, 60)) + np.array((20, 112, 112))


# In[265]:


square_bbox = [66, 86, 118, 230, 60, 172]
print("Sq bbox", square_bbox)


# In[266]:


# splits
splits = RandomSplitter(seed=42)(items)
print(f"Training: {len(splits[0])}, Valid: {len(splits[1])}")

# tfms
tfms = [PathToSitk(), IsotropicTfm(1), ToTensor3D(), CropBBox(square_bbox)] #, AddChannel()]

# tls
tls = TfmdLists(items, tfms, splits)

# dls
dls = tls.dataloaders(bs=2, after_batch=AddChannel())

# GPU
dls = dls.cuda()

# test get one batch
b = dls.one_batch()
print(type(b), b[0].shape, b[1].shape)


# # Metric
# 
# Linear combination of Dice and Cross Entropy

# In[250]:


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


# ipython nbconvert --to python  '6 - Dataloaders- NB - Simple-Copy1.ipynb'


# # Learner

# In[267]:


backbone = efficientnet_b0 #r3d_18 (pretrained?)
learn    = unet_learner_3d(dls, backbone, n_out=2)

# GPU
learn.model = learn.model.cuda()
learn = learn.to_fp16()


# In[269]:


# test:

#dls.device = "cpu"

start = time.time()

x,y = dls.one_batch()
#x,y = to_cpu(x), to_cpu(y)

pred = learn.model(x)
loss = learn.loss_func(pred, y)

elapsed = time.time() - start

print(f"Elapsed: {elapsed}")
print("Batch: x,y")
print(type(x), x.shape, x.dtype, "\n", type(y), y.shape, y.dtype)

print("Pred shape")
print(type(pred), pred.shape)

print("Loss")
print(loss)
print(learn.loss_func)


# In[ ]:


from fastai.callback.all import *
print(learn.show_training_loop())


# In[273]:


# learn.summary()


# # LR Finder

# In[270]:


# learn.lr_find()


# In[271]:


print("PRE learn.fit one cycle")
learn.fit_one_cycle(3, 0.01, wd = 1e-4)


# In[ ]:


print("unfreeze, learn 50")
# learn.unfreeze()
# learn.fit_one_cycle(50, 1e-3, wd = 1e-4)

