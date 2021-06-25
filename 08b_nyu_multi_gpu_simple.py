#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DATALOADER PARAMS
bs          = 20
nepochs     = 30
num_workers = 2

# PREPROCESS (Isotropic, PadResize)
iso       = 3
maxs      = [87, 90, 90]

# Train:Valid:Test = 60:20:20
train_pct, valid_pct, test_pct = .60, .20, .20


# In[2]:


# CHECK HARDWARE 

import os
import torch

gpu_count = torch.cuda.device_count()
cpu_count = os.cpu_count()
print("#GPU = {0:d}, #CPU = {1:d}".format(gpu_count, cpu_count))


# # Goal
# 
# Train hybrid OBELISK-NET/UNET. Tune batch size and presize HWD dimensions.
# - Preprocess: Smooth, intensity norm (N4 bias, hist bin matching)
# - Augmentations: flip, orientation, 10 deg
# - Thanks to: OBELISK, FAIMED3D
#     - https://github.com/mattiaspaul/OBELISK
#     -  https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# # Paths

# In[35]:


import os

# Paths to (1) code (2) data (3) saved models
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit/PitMRdata"
model_src   = "/gpfs/data/oermannlab/private_data/DeepPit/saved_models"

# UMich 
# code src: "/home/labcomputer/Desktop/Rachel"
# data src: "../../../../..//media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/rachel/"

deepPit_src = f"{code_src}/DeepPit"
obelisk_src = f"{code_src}/OBELISK"
label_src   = f"{data_src}/samir_labels"
ABIDE_src   = f"{data_src}/ABIDE"

# print
print("Folders in data src: ", end=""); print(*os.listdir(data_src), sep=", ")
print("Folders in label src (data w labels): ", end=""); print(*os.listdir(label_src), sep=", ")
print("Folders in ABIDE src (data wo labels) ", end=""); print(*os.listdir(ABIDE_src), sep=", ")


# # Imports

# In[36]:


# imports
from transforms import AddChannel, Iso, PadSz

# Utilities
import os
import sys
import time
import pickle
from pathlib import Path

# Input IO
import SimpleITK as sitk
import meshio

# Numpy and Pandas
import numpy as np
import pandas as pd
from pandas import DataFrame as DF

# Fastai + distributed training
from fastai import *
from fastai.torch_basics import *
from fastai.basics import *
from fastai.distributed import *

# PyTorch
from torchvision.models.video import r3d_18
from fastai.callback.all import SaveModelCallback
from torch import nn

# Obelisk
sys.path.append(deepPit_src)
sys.path.append(obelisk_src)

# OBELISK
from utils import *

# 3D extension to FastAI
# from faimed3d.all import *

# Helper functions
from helpers.preprocess import get_data_dict, paths2objs, folder2objs, seg2mask, mask2bbox, print_bbox, get_bbox_size, print_bbox_size
from helpers.general import sitk2np, np2sitk, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups
from helpers.viz import viz_axis


# # Data

# ## Paths

# ## Items 

# In[7]:


# Get data dict
data = {}
folders = os.listdir(label_src)
for folder in folders: data.update(get_data_dict(f"{label_src}/{folder}"))

# Convert data dict => items (path to MR, path to Segm tensor)
items = list(data.values())

# Split train/valid/test split
train_idxs, test_idxs = RandomSplitter(valid_pct=test_pct)(items)
train_items = [items[i] for i in train_idxs]
test_items  = [items[i] for i in test_idxs]

train_idxs, valid_idxs = RandomSplitter(valid_pct=valid_pct)(train_items)
train_items = [items[i] for i in train_idxs]
valid_items = [items[i] for i in valid_idxs]

# print
print(f"Total {len(items)} items in dataset.")
print(f"Train: {len(train_items)} items.")
print(f"Valid: {len(valid_items)} items.")
print(f"Test: {len(test_items)} items.")

# Save test idxs

# file name
model_time = time.ctime() # 'Mon Oct 18 13:35:29 2010'
model_name = f"iso_{iso}mm_pad_{maxs[0]}_{maxs[1]}_{maxs[2]}_bs_{bs}_test_sz_{len(test_items)}_epochs_{nepochs}_time_{model_time}"
print(f"Model name: {model_name}")

# save test set indices
with open(f"{model_src}/{model_name}_test_items.pkl", 'wb') as f:
    pickle.dump(list(test_items), f)


# In[8]:


# with open(f"{model_src}/{model_name}_test_items.pkl", 'rb') as f:
#     check_test_items = pickle.load(f)
#     print(check_test_items==test_items)
#     print(check_test_items[0])


# In[37]:


# # unique, for rapid prototyping

# # MR files: unique sz, sp, dir
# with open(f'{deepPit_src}/saved_metadata/unique_sz_sp_dir.pkl', 'rb') as f:
#     unique = pickle.load(f)

# # Create (MR path, Segm path) item from MR path
# def get_folder_name(s):
#     start = s.index("samir_labels/")
#     s = s[start + len("samir_labels/50373-50453/"):]
#     return s[0:s.index("/")]

# def change_prefix(s):
#     start = s.index("samir_labels/")
#     return f"{label_src}/{s[start+len('samir_labels/'):]}"

# # get unique
# unique = [(change_prefix(mr), data[get_folder_name(mr)][1]) for mr in unique]


# # Transforms
# 
# 1. Isotropic 3mm or Resize to 50x50x50 dimensions
# 2. Crop/Pad to common dimensions

# In[30]:


# # test

# tfms = [Iso(3)]
# tls = TfmdLists(unique, tfms)

# start = time.time()
# iso_szs = [mr.shape for mr,mk in tls]
# elapsed = time.time() - start

# print(f"Elapsed: {elapsed} s for {len(unique)} items.")


# In[31]:


# start = time.time()
# iso_szs = [mr.shape for mr,mk in tls]
# elapsed = time.time() - start

# print(f"Elapsed: {elapsed} s for {len(unique)} items.")


# In[32]:


# print(*[f"{get_folder_name(mr)}: {tuple(sz)}" for (mr,mk),sz in zip(unique, iso_szs)], sep="\n")


# In[33]:


# maxs = [int(x) for x in torch.max(torch.tensor(iso_szs), dim=0).values]
# print("Maxs: ", maxs)


# # Crop

# In[34]:


# # test
# iso_items = list(tls[0:2])

# # tfms
# pad_tfms = [PadSz(maxs)]

# # tls
# pad_tls = TfmdLists(iso_items, pad_tfms)

# pad_tls[0][0].shape, pad_tls[1][0].shape


# In[ ]:





# # Dataloaders
# 
# TODO augmentations.
# 
# - dset = tfms applied to items
# - splits into training/valid
# - bs

# In[46]:


# time it
start = time.time()

# splits
#splits = RandomSplitter(seed=42)(subset)
#print(f"Training: {len(splits[0])}, Valid: {len(splits[1])}")

# tfms
tfms = [Iso(3), PadSz(maxs)]

# tls
tls = TfmdLists(items, tfms, splits=(train_idxs, valid_idxs))

# dls
dls = tls.dataloaders(bs=bs, after_batch=AddChannel(), num_workers=num_workers)

# GPU
dls = dls.cuda()

# end timer
elapsed = time.time() - start
print(f"Elapsed time: {elapsed} s for {len(train_idxs) + len(valid_idxs)} items")

# test get one batch
b = dls.one_batch()
print(type(b), b[0].shape, b[1].shape)
print(len(dls.train), len(dls.valid))


# # Metric
# 
# Linear combination of Dice and Cross Entropy

# In[ ]:


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


# start = time.time()

# segs = torch.cat([tl[1] for tl in dls.train],0)
# print(segs.shape)

# elapsed = time.time() - start

# print(f"Elapsed time: {elapsed} s for {len(segs)} items")


# In[ ]:


# class_weight = torch.sqrt(1.0/(torch.bincount(segs.view(-1)).float()))
# class_weight = class_weight/class_weight.mean()
# class_weight[0] = 0.5
# np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
# print('inv sqrt class_weight',class_weight.data.cpu().numpy())


# In[ ]:


from utils import my_ohem


# In[ ]:


# pos_weight = torch.load("saved_metadata/class_weights.pt")
# class_weights = [0, pos_weight]

# # inv
# class_weights [1.0/x for x in class_wei]
# my_criterion = my_ohem(.25,[0, pos_weight]) #.cuda())#0.25 


# In[ ]:


def obelisk_loss_fn(predict, target): return my_criterion(F.log_softmax(predict,dim=1),target)


# In[ ]:


# ipython nbconvert --to python  '6 - Dataloaders- NB - Simple-Copy1.ipynb'


# # Learner

# In[ ]:


import gc

gc.collect()

torch.cuda.empty_cache()


# In[38]:


# OBELISK-NET from github
from models import obelisk_visceral, obeliskhybrid_visceral


# In[ ]:


full_res = maxs

learn = Learner(dls=dls,                 model=obeliskhybrid_visceral(num_labels=2, full_res=full_res),                 loss_func= loss, #DiceLoss(), #nn.CrossEntropyLoss(), \
                metrics = dice_score, \
                model_dir = model_src, \
                cbs = [SaveModelCallback(monitor='dice_score', fname=model_name, with_opt=True)])

# SaveModelCallback: model_dir = "./models", cbs = [SaveModelCallback(monitor='dice_score')]

# GPU
learn.model = learn.model.cuda()

#learn = learn.to_distributed(args.local_rank)


# In[ ]:


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

# In[ ]:


# learn.lr_find()


# In[ ]:


print("PRE learn.fit one cycle")
with learn.distrib_ctx():
    learn.fit_one_cycle(1, 3e-3, wd = 1e-4)


# In[ ]:


print("unfreeze, learn 50")
learn.unfreeze()
with learn.distrib_ctx():
    learn.fit_one_cycle(nepochs, 3e-3, wd = 1e-4)


# In[ ]:


# learn.save('iso_3mm_pad_87_90_90_subset_50_epochs_50')


# In[ ]:





# In[ ]:


# learn.lr_find()


# In[ ]:


# print("unfreeze, learn 50")
# learn.unfreeze()
# learn.fit_one_cycle(50, 1e-3, wd = 1e-4)


# In[ ]:





# In[ ]:





# In[ ]:


# testmask = torch.tensor([[[False, False, False], [False, False, False], [True, True, True]],
#                        [[False, False, False], [False, False, True], [True, True, True]],
#                        [[False, False, False], [False, False, False], [False, False, False]]])
# testmask


# In[ ]:


# testmaskN = np.array(testmask)
# testmaskN


# In[ ]:


# maskT = testmask.type(torch.BoolTensor)

# iT = torch.any(maskT, dim=(1,2))
# jT = torch.any(maskT, dim=(0,2))
# kT = torch.any(maskT, dim=(0,1))

# iminT, imaxT = torch.where(iT)[0][[0, -1]]
# jminT, jmaxT = torch.where(jT)[0][[0, -1]]
# kminT, kmaxT = torch.where(kT)[0][[0, -1]]


# In[ ]:


# maskN = np.array(testmask).astype(bool)
    
# iN = np.any(maskN, axis=(1, 2))
# jN = np.any(maskN, axis=(0, 2))
# kN = np.any(maskN, axis=(0, 1))

# iminN, imaxN = np.where(iN)[0][[0, -1]]
# jminN, jmaxN = np.where(jN)[0][[0, -1]]
# kminN, kmaxN = np.where(kN)[0][[0, -1]]


# In[ ]:


# maskT.shape, maskN.shape


# In[ ]:


# print(iT)
# print(jT)
# print(kT)
# print([x for x in (iminT, imaxT, jminT, jmaxT, kminT, kmaxT)])


# In[ ]:


# print(iN)
# print(jN)
# print(kN)
# print([int(x) for x in (iminN, imaxN, jminN, jmaxN, kminN, kmaxN)])


# In[ ]:


#     def torch_mask2bbox(mask):
#         mask = mask.type(torch.BoolTensor)

#         i = torch.any(mask, dim=0)
#         j = torch.any(mask, dim=1)
#         k = torch.any(mask, dim=2)

#         imin, imax = torch.where(i)[0][[0, -1]]
#         jmin, jmax = torch.where(j)[0][[0, -1]]
#         kmin, kmax = torch.where(k)[0][[0, -1]]

#         # inclusive idxs
#         return imin, imax+1, jmin, jmax+1, kmin, kmax+1

