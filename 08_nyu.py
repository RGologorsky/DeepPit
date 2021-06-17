# # Goal
# 
# This notebook is converted to .py for the purpose of training a **hybrid OBELISK-NET/UNET** segmentation model. A key concern is **memory usage**, i.e. tuning the batch size and presize HWD dimensions.
# 
# - TODO: Preprocess: Smooth, intensity norm (N4 bias, hist bin matching)
# - TODO Augmentations: flip, orientation, 10 deg
# 
# **With gratitude to**:
# - https://github.com/mattiaspaul/OBELISK
# -  https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# # Imports

# In[2]:

print("CUDA: ", torch.cuda.is_available())

import sys
sys.path.append('/gpfs/home/gologr01/DeepPit')
sys.path.append('/gpfs/home/gologr01/OBELISK')


# imports
# from transforms import AddChannel, Iso, Pad

# Utilities
import os
import time
import pickle
from pathlib import Path

# Fastai
from fastai import *
from fastai.torch_basics import *
from fastai.basics import *

# PyTorch
from torchvision.models.video import r3d_18
from fastai.callback.all import SaveModelCallback
from torch import nn

# 3D extension to FastAI
# from faimed3d.all import *

# Input IO
import SimpleITK as sitk
import meshio

# Numpy and Pandas
import numpy as np
import pandas as pd
from pandas import DataFrame as DF

# Helper functions
from helpers.preprocess import get_data_dict, paths2objs, folder2objs, seg2mask, mask2bbox, print_bbox, get_bbox_size, print_bbox_size
from helpers.general import sitk2np, np2sitk, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups
from helpers.viz import viz_axis


# In[31]:


class AddChannel(DisplayedTransform):
    "Adds Channels dims, in case they went missing"
    split_idx,order = None, 99
        
    def encodes(self, x:Tensor):
        if x.ndim == 3: x = x.unsqueeze(0)
        if x.ndim == 4: x = x.unsqueeze(1)
        return x
    
class Iso(ItemTransform):
    
    def __init__(self, new_sp = 3):
        self.new_sp = new_sp
        
    def encodes(self, x):
        # get sitk objs
        im_path, segm_path = x
        mr = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im = torch.swapaxes(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
        mk = torch.load(f"{str(Path(segm_path).parent)}/seg.pt").float()

        # resize so isotropic spacing
        orig_sp = mr.GetSpacing()
        orig_sz = mr.GetSize()
        new_sz = [int(round(osz*ospc/self.new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

        while im.ndim < 5: 
            im = im.unsqueeze(0)
            mk = mk.unsqueeze(0)

        return F.interpolate(im, size = new_sz, mode = 'trilinear', align_corners=False).squeeze(),                 
                F.interpolate(mk, size = new_sz, mode = 'nearest').squeeze().long()    

# pad to new size
class PadSz(Transform):
    def __init__(self, new_sz):
        self.new_sz = new_sz
    
    def encodes(self, arr):
        pad = [x-y for x,y in zip(self.new_sz, arr.shape)]
        pad = [a for amt in pad for a in (amt//2, amt-amt//2)]
        pad.reverse()
        
        return F.pad(arr, pad, mode='constant', value=0) 


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

# 1. Source = path to labels (segmentation)
# 2. data dict[foldername] = (path to MR, path to Segm tensor)
#     
# Special subsets:
# 1. *training*: small subset of all labelled items (quick epoch w/ 100 instead of 335 items).
# 2. *unique*: subset of items with unique size, spacing, and orientation (quickly evaluate resize vs. istropic)

# In[ ]:


subset_size = 15


# In[5]:


test_pct  = 1 - float(subset_size)/335
bs        = 2
nepochs   = 50
num_workers = 1

iso       = 3
maxs      = [87, 90, 90]



# In[ ]:


import time
model_time = time.ctime() # 'Mon Oct 18 13:35:29 2010'
print(f"Time: {model_time}")


# In[6]:


# cognizant of diff file paths
todd_prefix = "../../../../..//media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/rachel/"
olab_prefix = "/gpfs/data/oermannlab/private_data/DeepPit/"


# In[ ]:


# process for new prefix
def change_prefix(fn, old_prefix=todd_prefix, new_prefix=olab_prefix):
    return new_prefix + fn[len(old_prefix):]

# Get path to my data on 4 TB HD
hd  = "/gpfs/data/oermannlab/private_data/DeepPit/PitMRdata"
src = hd

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

# MR files: unique sz, sp, dir
with open('saved_metadata/unique_sz_sp_dir.pkl', 'rb') as f:
    unique = pickle.load(f)

# Create (MR path, Segm path) item from MR path
def get_folder_name(s):
    start = s.index("samir_labels/")
    s = s[start + len("samir_labels/50373-50453/"):]
    return s[0:s.index("/")]

# get unique
unique = [(change_prefix(mr), data[get_folder_name(mr)][1]) for mr in unique]

# subset
subset_idxs, test_idxs = RandomSplitter(valid_pct=test_pct)(items)
subset = [items[i] for i in subset_idxs]
test   = [items[i] for i in test_idxs]

# print
print(f"Total {len(items)} items in dataset.")
print(f"Training subset of {len(subset)} items.")
print(f"Test subset of {len(test)} items.")

# model name
model_name = f"iso_{iso}mm_pad_{maxs[0]}_{maxs[1]}_{maxs[2]}_bs_{bs}_subset_{len(subset)}_epochs_{nepochs}_time_{model_time}"
print(f"Model name: {model_name}")

# save test set indices
with open(f'model_test_sets/{model_name}_test_items.pkl', 'wb') as f:
    pickle.dump(list(test), f)
    
# print
print(f"Total {len(items)} items in dataset.")
print(f"Training subset of {len(subset)} items.")
print(f"Unique subset of {len(unique)} items.")


# In[8]:


# model name
model_name = f"iso_3mm_pad_87_90_90_bs_{bs}_subset_{len(subset)}_epochs_{nepochs}_time_{model_time}\"\n",
print(f"Model name: {model_name}\")\n",

# save test set indices\n",
with open(f'model_test_sets/{model_name}_test_items.pkl', 'wb') as f:
    pickle.dump(list(test), f)
      
# with open(f"model_test_sets/{model_name}_test_items.pkl", 'rb') as f:
#     test = pickle.load(f)
# print(test[0]), print(len(test))


# # Transforms
# 
# 1. Isotropic 3mm or Resize to 50x50x50 dimensions
# 2. Crop/Pad to common dimensions

# In[13]:


# test

tfms = [Iso(3)]
tls = TfmdLists(unique, tfms)

start = time.time()
iso_szs = [mr.shape for mr,mk in tls]
elapsed = time.time() - start

print(f"Iso 3: Elapsed: {elapsed} s for {len(unique)} items.")


# In[14]:


start = time.time()
iso_szs = [mr.shape for mr,mk in tls]
elapsed = time.time() - start

print(f"Shapes: Elapsed: {elapsed} s for {len(unique)} items.")


# In[15]:


print(*[f"{get_folder_name(mr)}: {tuple(sz)}" for (mr,mk),sz in zip(unique, iso_szs)], sep="\n")


# In[16]:


maxs = [int(x) for x in torch.max(torch.tensor(iso_szs), dim=0).values]
print("Maxs: ", maxs)


# # Crop

# In[32]:


iso_items = list(tls[0:2])


# In[33]:


# test

# tfms
pad_tfms = [PadSz(maxs)]

# tls
pad_tls = TfmdLists(iso_items, pad_tfms)


# In[34]:


print("PAD tls shape: ", pad_tls[0][0].shape, pad_tls[1][0].shape)


# # Dataloaders
# 
# TODO augmentations.
# 
# - dset = tfms applied to items
# - splits into training/valid
# - bs

# In[37]:


# time it
start = time.time()

# splits
splits = RandomSplitter(seed=42)(subset)
print(f"Training: {len(splits[0])}, Valid: {len(splits[1])}")

# tfms
tfms = [Iso(3), PadSz(maxs)]

# tls
tls = TfmdLists(items, tfms, splits=splits)

# dls
dls = tls.dataloaders(bs=bs, after_batch=AddChannel(), num_workers=num_workers)

# GPU
dls = dls.cuda()

# end timer
elapsed = time.time() - start
print(f"Elapsed time: {elapsed} s for {len(subset)} items")

# test get one batch
b = dls.one_batch()
print(type(b), b[0].shape, b[1].shape)
print(len(dls.train), len(dls.valid))


# # Metric
# 
# Linear combination of Dice and Cross Entropy

# In[38]:


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


# ## OBELISK

# In[87]:


from utils import *


# In[43]:


start = time.time()

segs = torch.cat([tl[1] for tl in dls.train],0)
print(segs.shape)

elapsed = time.time() - start

print(f"Elapsed time: {elapsed} s for {len(segs)} items")


# In[44]:


class_weight = torch.sqrt(1.0/(torch.bincount(segs.view(-1)).float()))
class_weight = class_weight/class_weight.mean()
class_weight[0] = 0.5
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
print('inv sqrt class_weight',class_weight.data.cpu().numpy())

# In[47]:


from utils import my_ohem


# In[56]:


# In[59]:


my_criterion = my_ohem(.25,class_weight).cuda() #)#0.25 


# In[60]:


def obelisk_loss_fn(predict, target): return my_criterion(F.log_softmax(predict,dim=1),target)


# In[61]:


# ipython nbconvert --to python  '6 - Dataloaders- NB - Simple-Copy1.ipynb'


# # Learner

# In[62]:


import gc

gc.collect()

torch.cuda.empty_cache()


# In[63]:


# OBELISK-NET from github
from models import obelisk_visceral, obeliskhybrid_visceral


# In[64]:


full_res = maxs

learn = Learner(dls=dls,                 
                model=obeliskhybrid_visceral(num_labels=2, full_res=full_res),                 
                loss_func= obelisk_loss_fn, #DiceLoss(), #nn.CrossEntropyLoss(), \
                metrics = dice_score, \
                model_dir = "./models", \
                cbs = [SaveModelCallback(monitor='dice_score'), fname=model_name, with_opt=True])

# SaveModelCallback: model_dir = "./models", cbs = [SaveModelCallback(monitor='dice_score')]

# GPU
learn.model = learn.model.cuda()


# In[65]:


# test:

#dls.device = "cpu"

start = time.time()

x,y = dls.one_batch()
#x,y = to_cpu(x), to_cpu(y)

pred = learn.model(x)
loss = learn.loss_func(pred, y)

elapsed = time.time() - start

print(f"Elapsed: {elapsed} s")
print("Batch: x,y")
print(type(x), x.shape, x.dtype, "\n", type(y), y.shape, y.dtype)

print("Pred shape")
print(type(pred), pred.shape, pred.dtype)

print("Loss")
print(loss)
print(learn.loss_func)


# # LR Finder

# In[236]:


#learn.lr_find()


# In[66]:


print("PRE learn.fit one cycle")
learn.fit_one_cycle(1, 3e-3, wd = 1e-4)


# In[67]:


print("unfreeze, learn 50")
learn.unfreeze()
learn.fit_one_cycle(nepochs, 3e-3, wd = 1e-4)


# In[86]:


# learn.save('iso_3mm_pad_87_90_90_subset_50_epochs_50')


# In[72]:


# learn.lr_find()


# In[ ]:


# print("unfreeze, learn 50")
# learn.unfreeze()
# learn.fit_one_cycle(50, 1e-3, wd = 1e-4)
