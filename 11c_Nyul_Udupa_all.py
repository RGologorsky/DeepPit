#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Nyul-Udupa histogram rescaling
# 
# 1. compute landmarks for all
# 2. standard scale = avg landmarks (total sum div by total #inputs)

# # Imports

# In[1]:


import os

# Paths to (1) code (2) data (3) saved models (4) where to save Nyul-Udupa landmarks
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit/PitMRdata"
model_src   = "/gpfs/data/oermannlab/private_data/DeepPit/saved_models"
save_src    = "/gpfs/data/oermannlab/private_data/DeepPit/saved_landmarks/ABIDE"

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


# In[2]:


from fastai.vision.core import *


# In[3]:


# imports
from transforms import AddChannel, Iso, PadSz

# Utilities
import os
import sys
import time
import pickle
from pathlib import Path

# regex
from re import search

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
from models import obelisk_visceral, obeliskhybrid_visceral

# 3D extension to FastAI
# from faimed3d.all import *

# Helper functions
from helpers.preprocess import get_data_dict, paths2objs, folder2objs, seg2mask, mask2bbox, print_bbox, get_bbox_size, print_bbox_size
from helpers.general import sitk2np, np2sitk, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups
from helpers.viz import viz_axis


# # MR data

# In[18]:


# Load fnames from .txt
with open(f"{deepPit_src}/saved_metadata/ABIDE.txt", 'rb') as f:
    fnames = pickle.load(f)
print(len(fnames), fnames[0])

def change_src(s, old_src="../../../mnt/d/PitMRdata", new_src=data_src): return new_src + s[len(old_src):] 
fnames = [change_src(f) for f in fnames]
print(len(fnames), fnames[0])

# exclude PAD (not .nii files)
fnames_no_pad = [f for f in fnames if not "PAD" in f]
print(len(fnames_no_pad), fnames_no_pad[0])


# In[19]:


# get corrected N4
fnames_no_pad = [glob.glob(f"{f}/*corrected_n4.nii")[0] for f in fnames_no_pad]


# # Get chunk

# In[20]:


import os
taskid = int(os.getenv('SLURM_ARRAY_TASK_ID') or 0)
   
n_total = len(fnames_no_pad)

chunk_len = 20    
chunks    = [range(i,min(i+chunk_len, n_total)) for i in range(0, n_total, chunk_len)]

print(f"N_chunks = {len(chunks)}")
# print(f"Array Task ID: {taskid}")
# print(f"Array ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
# print(f"Job ID: {os.getenv('SLURM_JOB_ID')}")
#print(*chunks, sep="\n")

task_chunk = chunks[taskid]


# # Transform
# 
# ## from FAIMED3D 02_preprocessing

# In[21]:


# from FAIMED3D 02_preprocessing


# Piecewise linear histogram matching
# [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081, 1999.
# 
# [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold, D. L. Collins, and T. Arbel, “Evaluating intensity normalization on MRIs of human brain with multiple sclerosis,” Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
# 
# Implementation adapted from: https://github.com/jcreinhold/intensity-normalization, ported to pytorch (no use of numpy works in cuda).
# 
# In contrast to hist_scaled, the piecewise linear histogram matching need pre-specified values for new scale and landmarks. It should be used to normalize a whole dataset.

# In[22]:


from torch import Tensor


# In[23]:


def get_percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (float).

    This function is twice as fast as torch.quantile and has no size limitations
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k)[0].item()

    return result


# In[24]:


def get_landmarks(t: torch.Tensor, percentiles: torch.Tensor)->torch.Tensor:
    """
    Returns the input's landmarks.

    :param t (torch.Tensor): Input tensor.
    :param percentiles (torch.Tensor): Peraentiles to calculate landmarks for.
    :return: Resulting landmarks (torch.tensor).
    """
    return tensor([get_percentile(t, perc.item()) for perc in percentiles])


# In[30]:


def find_sum_landmarks(inputs, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images
    Args:
        inputs (list or L): set of TensorDicom3D objects which are to be normalized
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)
    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    percs = torch.cat([torch.tensor([i_min]),
                       torch.arange(l_percentile, u_percentile+1, step),
                       torch.tensor([i_max])], dim=0)
    standard_scale = torch.zeros(len(percs))

    for input_image in inputs:
        mask_data = input_image > input_image.mean()
        masked = input_image[mask_data]
        landmarks = get_landmarks(masked, percs)
        min_p = get_percentile(masked, i_min)
        max_p = get_percentile(masked, i_max)
        new_landmarks = landmarks.interp_1d(torch.FloatTensor([i_s_min, i_s_max]),
                                            torch.FloatTensor([min_p, max_p]))
        standard_scale += new_landmarks
    #standard_scale = standard_scale / len(inputs)
    return standard_scale, percs


# In[26]:


def path2tensor(mr_path):
    mr = sitk.ReadImage(mr_path, sitk.sitkFloat32)
    return torch.transpose(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)


# # Process

# In[28]:


nii_files     = [fnames_no_pad[i] for i in task_chunk]


# In[ ]:


landmarks_sum, percs = find_sum_landmarks([path2tensor(f) for f in nii_files])


# In[34]:


# write standard scale
save_loc = "/gpfs/data/oermannlab/private_data/DeepPit/saved_landmarks/ABIDE"
torch.save(landmarks_sum, f"{save_loc}/{taskid}_landmarks_sum.pt")
torch.save(torch.Tensor(len(nii_files)), f"{save_loc}/{taskid}_nfiles.pt")

d = {"nii_files": nii_files, "percs": percs}
with open(f'{save_loc}/{taskid}_info.pickle', 'wb') as handle:
    pickle.dump(d, handle)


# In[ ]:




