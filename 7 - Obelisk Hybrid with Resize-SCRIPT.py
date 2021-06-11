#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# This notebook is converted to .py for the purpose of training a **hybrid OBELISK-NET/UNET** segmentation model. A key concern is **memory usage**, i.e. tuning the batch size and presize HWD dimensions.
# 
# **Dataset**:
# - Training Examples = 335:
# - Input  = (MR series path, Segmentation obj path) in samir_labels (ABIDE dataset)
# - Output = binary mask representing the sella turcica ROI.
# 
# **Augmentations**:
# - Resize to 3mm isotropic
# - Smooth inputs
# 
# **To Do**:
# - TODO Augmentations: flip, orientation
# - TODO Intensity normalization: N4 bias correction, hist bin matching, tissue intensity,
# 
# **With gratitude to**:
# - https://github.com/mattiaspaul/OBELISK
# -  https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# # Imports

# In[2]:


# imports

# Utilities
import os
import time
import pickle
from pathlib import Path

# Fastai + PyTorch
from fastai import *
from torchvision.models.video import r3d_18
from fastai.callback.all import SaveModelCallback
from torch import nn

# 3D extension to FastAI
from faimed3d.all import *

# Input IO
import SimpleITK as sitk
import meshio

# Numpy and Pandas
import numpy as np
import pandas as pd
from pandas import DataFrame as DF

# Helper functions
from helpers_preprocess import get_data_dict, paths2objs, folder2objs, seg2mask, mask2bbox, print_bbox, get_bbox_size, print_bbox_size
from helpers_general import sitk2np, np2sitk, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups
from helpers_viz import viz_axis


# # GPU Stats

# In[3]:


# clear cache
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# print GPU stats
import GPUtil as GPU
GPUs = GPU.getGPUs()
for gpu in GPUs:
    print("GPU {0:20s} RAM Free: {1:.0f}MB | Used: {2:.0f}MB | Util {3:3.0f}% | Total {4:.0f}MB".format(gpu.name, gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))


# # Distributed Training

# In[4]:


# from fastai.distributed import *
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()
# torch.cuda.set_device(args.local_rank)
# torch.distributed.init_process_group(backend='nccl', init_method='env://')


# # Data

# 1. Set source = path to where labelled data is stored (on HD, etc)
# 2. Load items into dictionary:
#     - key = foldername
#     - value = (path to MR series, path to Segmentation obj)
#     
# Special subsets:
# 1. *training*: small subset of all labelled items (quick epoch w/ 100 instead of 335 items).
# 2. *unique*: subset of items with unique size, spacing, and orientation (quickly evaluate resize vs. istropic)

# In[5]:


test_pct  = 1 - 300.0/335
bs        = 2
maxs      = [87, 90, 90]
nepochs   = 50

num_workers = 1


# In[20]:


import time
model_time = time.ctime() # 'Mon Oct 18 13:35:29 2010'
print(f"Time: {model_time}")


# In[24]:


# Get path to my data on 4 TB HD
hd  = "../" * 5 + "/media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a"
src = hd + "/home/rachel/PitMRdata"

# labelled train data
train_src = src + "/samir_labels"

# print
print("Folders in source path: ", end=""); print(*os.listdir(src), sep=", ")
print("Folders in train path: ", end=""); print(*os.listdir(train_src), sep=", ")

# get data
data = {}
folders = os.listdir(train_src)
for folder in folders: data.update(get_data_dict(f"{train_src}/{folder}"))

# all items
items = list(data.values())

# subset
subset_idxs, test_idxs = RandomSplitter(valid_pct=test_pct)(items)
subset = [items[i] for i in subset_idxs]
test   = [items[i] for i in test_idxs]

# print
print(f"Total {len(items)} items in dataset.")
print(f"Training subset of {len(subset)} items.")
print(f"Test subset of {len(test)} items.")

# model name
model_name = f"iso_3mm_pad_87_90_90_bs_{bs}_subset_{len(subset)}_epochs_{nepochs}_time_{model_time}"
print(f"Model name: {model_name}")

# save test set indices
with open(f'model_test_sets/{model_name}_test_items.pkl', 'wb') as f:
    pickle.dump(list(test), f)


# In[25]:


# with open(f"model_test_sets/{model_name}_test_items.pkl", 'rb') as f:
#     test = pickle.load(f)
# print(test[0]), print(len(test))


# # Transforms
# 
# 1. Isotropic 3mm or Resize to 50x50x50 dimensions
# 2. Crop/Pad to common dimensions

# In[51]:


class DoAll(ItemTransform):
    
    def __init__(self, new_sp = 3):
        self.new_sp = new_sp
        
    def encodes(self, x):
        # get sitk objs
        im_path, segm_path = x
        folder  = Path(segm_path).parent.name
        ras_adj = int(folder) in range(50455, 50464)

        mr         = sitk.ReadImage(im_path, sitk.sitkFloat32)
        segm       = meshio.read(segm_path)
        mask_arr   = seg2mask(mr, segm, ras_adj)

        # resize so isotropic spacing
        orig_sp = mr.GetSpacing()
        orig_sz = mr.GetSize()
        new_sz = [int(round(osz*ospc/self.new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

        im = torch.swapaxes(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
        mk = torch.tensor(mask_arr).float()

        while im.ndim < 5: 
            im = im.unsqueeze(0)
            mk = mk.unsqueeze(0)

        return F.interpolate(im, size = new_sz, mode = 'trilinear', align_corners=False).squeeze(),                 F.interpolate(mk, size = new_sz, mode = 'nearest').squeeze().long()


# # Crop

# In[71]:


# crop by coords
class PadResize(Transform):
    def __init__(self, new_sz):
        self.new_sz = new_sz
    
    def encodes(self, arr):
        pad = [x-y for x,y in zip(self.new_sz, arr.shape)]
        pad = [a for amt in pad for a in (amt//2, amt-amt//2)]
        pad.reverse()
        
        return F.pad(arr, pad, mode='constant', value=0)


# # Dataloaders
# 
# TODO augmentations.
# 
# - dset = tfms applied to items
# - splits into training/valid
# - bs

# In[76]:


# time it
start = time.time()

# splits
splits = RandomSplitter(seed=42)(subset)
print(f"Training: {len(splits[0])}, Valid: {len(splits[1])}")

# tfms
tfms = [DoAll(3), PadResize(maxs)]

# tls
tls = TfmdLists(items, tfms, splits=splits)

# dls
dls = tls.dataloaders(bs=bs, after_batch=AddChannel(), num_workers=num_workers)

# GPU
dls = dls.cuda()

# end timer
elapsed = time.time() - start
print(f"Elapsed time: {elapsed} s for {len(subset)} items, bs = {bs}")

# test get one batch
# b = dls.one_batch()
# print(type(b), b[0].shape, b[1].shape)
# print(len(dls.train), len(dls.valid))


# # Metric
# 
# Linear combination of Dice and Cross Entropy

# In[77]:


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


# In[78]:


# ipython nbconvert --to python  '6 - Dataloaders- NB - Simple-Copy1.ipynb'


# # Learner

# In[79]:


import gc

gc.collect()

torch.cuda.empty_cache()


# In[81]:


# OBELISK-NET from github
import sys
sys.path.append('/home/labcomputer/Desktop/Rachel/OBELISK')
from models import obelisk_visceral, obeliskhybrid_visceral


# In[ ]:


full_res = maxs

learn = Learner(dls=dls,                 model=obeliskhybrid_visceral(num_labels=2, full_res=full_res),                 loss_func= DiceLoss(), #nn.CrossEntropyLoss(), \
                metrics = dice_score, \
                model_dir = "./models", cbs = [SaveModelCallback(monitor='dice_score', fname=model_name, with_opt=True)])

# SaveModelCallback: model_dir = "./models", cbs = [SaveModelCallback(monitor='dice_score')]


# In[1]:





# In[ ]:


# load saved model
# learn.model_dir = ""
# learn = learn.load('models/iso_3mm_pad_87_90_90_bs_2_subset_168_epochs_50_time_Thu Jun 10 19:23:40 2021')


# In[82]:


# GPU
learn.model = learn.model.cuda()


# In[83]:


# # test:

# #dls.device = "cpu"

# start = time.time()

# x,y = dls.one_batch()
# #x,y = to_cpu(x), to_cpu(y)

# pred = learn.model(x)
# loss = learn.loss_func(pred, y)

# elapsed = time.time() - start

# print(f"Elapsed: {elapsed} s")
# print("Batch: x,y")
# print(type(x), x.shape, x.dtype, "\n", type(y), y.shape, y.dtype)

# print("Pred shape")
# print(type(pred), pred.shape, pred.dtype)

# print("Loss")
# print(loss)
# print(learn.loss_func)


# # LR Finder

# In[236]:


#learn.lr_find()


# In[84]:


print("PRE learn.fit one cycle")
learn.fit_one_cycle(1, 3e-3, wd = 1e-4)


# In[ ]:


print("unfreeze, learn 50")
learn.unfreeze()
learn.fit_one_cycle(nepochs, 3e-3, wd = 1e-4)


# In[72]:


# learn.lr_find()

