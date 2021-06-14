#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# This notebook explores the distribution of bounding box ROIs post-alignment. This range of values is used to inform the hyperparameter testing range (the optimal amount to pad the bounding box for cascaded alignment).

# # Imports

# In[1]:


# imports

import faulthandler
faulthandler.enable()

import os, sys, time
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from pandas import DataFrame as DF

import SimpleITK as sitk

from helpers_general import sitk2np, mask2sitk, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups
from helpers_preprocess import mask2bbox, print_bbox, get_bbox_size, print_bbox_size, get_data_dict, folder2objs,                                 threshold_based_crop, get_reference_frame, resample2reference

from helpers_metrics import compute_dice_coefficient, compute_coverage_coefficient
from helpers_viz import viz_axis


# In[2]:


# auto-reload when local helper fns change
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# get_ipython().run_line_magic('matplotlib', 'inline')


# # Data
# 
# - Load data from folder (train_path)
# - Crop to foreground
# - Get standard reference domain
# - Resample to sample reference domain

# In[3]:


# Data path

PROJ_PATH = "."

# Folders containing MR train data
train_path = f"{PROJ_PATH}/train_data/train_data"
train_data_dict = get_data_dict(train_path)

# print train data dict
print(f"Train data folders: {numbers2groups(sorted([int(x) for x in os.listdir(train_path)]))}")
print(f"Training data (size {len(train_data_dict)}): key = train folder, value = full path to (segm obj, nii file)\n")


# In[4]:


folders     = sorted([int(x) for x in os.listdir(train_path)])
nii_paths   = [train_data_dict[str(folder)][1] for folder in folders]


# In[5]:


def get_img_data(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return  reader.GetSize(), reader.GetSpacing() #reader.GetDirection()

all_img_data = [get_img_data(path) for path in nii_paths]

def get_reference_frame(all_img_data):
    img_data = all_img_data
    
    dimension = 3 # 3D MRs
    pixel_id = 2 # 16-bit signed integer

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    for img_sz, img_spc in img_data:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx else mx                                       for sz, spc, mx in zip(img_sz, img_spc, reference_physical_size)]

    print(reference_physical_size)
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

reference_frame = get_reference_frame(all_img_data)
reference_image, reference_center = get_reference_image(reference_frame)
print_sitk_info(reference_image)


# In[6]:


# Define which folders to load
obj_range = lrange(50002, 50017+1)


# In[7]:


def idx2obj(idx):
    # load obj => crop obj => resample2reference
    # 10 MRs labelled by Dr. Hollon, need nii LPS=>RAS adjustment for the mask
    ras_range = range(50455, 50464+1)
    return resample2reference(*threshold_based_crop(*folder2objs(str(idx), train_data_dict, idx in ras_range)),
                             reference_image, reference_center)


# ### Align
# 
# Align a sample of 10 pairs of images.

# In[13]:


n_objs = len(obj_range)
n_pairs = 10

print(f"n_objs: {n_objs}")
print(f"n_pairs: {n_pairs}")


# In[15]:


from itertools import combinations
idxs = list(combinations(obj_range,2))
print(len(idxs))
print(idxs[:5])


# In[16]:


from helpers_predict import atlas2pred


# In[17]:


# from IPython.display import Javascript
# display(Javascript('IPython.notebook.execute_cells_above()'))


# In[18]:


# store the img, mask arrs
memo = {}

def np2rs(arr, reference_frame = reference_frame):
    # check
    _, _, reference_origin, reference_spacing, reference_direction, _ = reference_frame
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(reference_origin)
    image.SetSpacing(reference_spacing)
    image.SetDirection(reference_direction)
    return image

# mask2bbox(sitk2np(atlas2pred(input_obj, atlas_obj, atlas_mask_obj)))
def idxs2pred_memo_all(input_idx, atlas_idx):
    
    # objs = img_obj, mask_obj
    
    if input_idx in memo:
        input_obj      = np2rs(memo[input_idx][0])
        input_mask_obj = np2rs(memo[input_idx][1])
    else:
        input_obj, input_mask_obj = idx2obj(input_idx)        
        # store (img arr, mask arr) in memo
        memo[input_idx] = sitk.GetArrayFromImage(input_obj),                           sitk.GetArrayFromImage(input_mask_obj)
        
    if atlas_idx in memo:
        atlas_obj      = np2rs(memo[atlas_idx][0])
        atlas_mask_obj = np2rs(memo[atlas_idx][1])
    else:
        atlas_obj, atlas_mask_obj = idx2obj(atlas_idx)
        memo[atlas_idx] = sitk.GetArrayFromImage(atlas_obj),                           sitk.GetArrayFromImage(atlas_mask_obj)
    
    # ground truth = input mask obj
    gt_bbox   = mask2bbox(sitk2np(input_mask_obj))
    
    # predicted = atlas2pred mask obj
    pred_mask_obj = atlas2pred(input_obj, atlas_obj, atlas_mask_obj)
    pred_bbox = mask2bbox(sitk2np(pred_mask_obj))
   
    return pred_bbox, gt_bbox

# mask2bbox(sitk2np(atlas2pred(input_obj, atlas_obj, atlas_mask_obj)))
def idxs2pred_bbox(input_idx, atlas_idx):
    
    # objs = img_obj, mask_obj
    input_obj, input_mask_obj = idx2obj(input_idx)  
    atlas_obj, atlas_mask_obj = idx2obj(atlas_idx)
    
    # ground truth = input mask obj
    gt_bbox   = mask2bbox(sitk2np(input_mask_obj))
    
    # predicted = atlas2pred mask obj
    pred_mask_obj = atlas2pred(input_obj, atlas_obj, atlas_mask_obj)
    pred_bbox = mask2bbox(sitk2np(pred_mask_obj))
   
    # store bbox's in memo
    for idx,bbox in ((input_idx, gt_bbox), (atlas_idx, mask2bbox(sitk2np(atlas_mask_obj)))):
        if idx not in memo: memo[idx] = bbox

    memo[(input_idx, atlas_idx)] = pred_bbox
    
    return pred_bbox, gt_bbox


# In[27]:


# Full size: (191, 268, 268) 
# Half size: (85, 134, 134)

imin_range = np.arange(10, 90, step=10)
imax_range = np.arange(10, 90, step=10)

jmin_range = np.arange(15, 125, step=10)
jmax_range = np.arange(15, 125, step=10)

kmin_range = np.arange(15,125, step=10)
kmax_range = np.arange(15,125, step=10)


# In[28]:


imin_range = np.arange(10, 50, step=20)
imax_range = np.arange(10, 50, step=20)

jmin_range = np.arange(15, 75, step=30)
jmax_range = np.arange(15, 75, step=30)

kmin_range = np.arange(15,75, step=30)
kmax_range = np.arange(15,75, step=30)


# In[29]:


pad_ranges = (imin_range, imax_range, jmin_range, jmax_range, kmin_range, kmax_range)
for r in pad_ranges:
    print(f"Len {len(r):2}: ", r)

from functools import reduce
product = reduce((lambda x, y: x * y), [len(r) for r in pad_ranges])
print(f"Total No. Parameter Combinations: {product}.")


# In[30]:


# store the img, mask arrs
csc_memo = {}
csc_errors = {}

def bbox2diff(gt_bbox, pred_bbox): 
    diff = (x1-x2 for x1,x2 in zip(gt_bbox, pred_bbox))
    cols = "delta_imin", "delta_imax", "delta_jmin", "delta_jmax", "delta_kmin", "delta_kmax"
    return dict(zip(cols, diff))

def crop2roi(objs, bbox_coords, mult_factor=1):
    
    imin, imax, jmin, jmax, kmin, kmax = bbox_coords    
    sizes = [sz*mult_factor for sz in (imax-imin, jmax-jmin, kmax-kmin)]
    pads  = [halve(x) for x in sizes]
    
    # HACKY
    #pads = [30, 70, 70]
    
    imin_pad, jmin_pad, kmin_pad = [max(0, m-pad) for m,pad in zip((imin, jmin, kmin), pads)]
    imax_pad, jmax_pad, kmax_pad = [min(sz, m+pad) for m,pad,sz in zip((imax, jmax, kmax), pads, objs[0].GetSize())]
    
    
    return      (*[o[imin_pad:imax_pad, jmin_pad:jmax_pad, kmin_pad:kmax_pad] for o in objs],                 (imin_pad, imax_pad, jmin_pad, jmax_pad, kmin_pad, kmax_pad))

# mask2bbox(sitk2np(atlas2pred(input_obj, atlas_obj, atlas_mask_obj)))
def idxs2csc_bbox(input_idx, atlas_idx, pad_amts):
    
    # pad dict
    pad_cols    = "pad_imin", "pad_imax", "pad_jmin", "pad_jmax", "pad_kmin", "pad_kmax"
    pad_dict    = dict(zip(pad_cols, pad_amts)) 
    
    print("Getting obj")
    
    # objs = img_obj, mask_obj
    input_obj, input_mask_obj = idx2obj(input_idx)  
    atlas_obj, atlas_mask_obj = idx2obj(atlas_idx)
    
    print("Loaded obj")
    
    # ground truth = input mask obj
    gt_mask_arr   = sitk2np(input_mask_obj).astype(bool)
    gt_bbox       = mask2bbox(gt_mask_arr)
    
    print("Getting pred")
    
    # predicted = atlas2pred mask obj
    pred_mask_arr = sitk2np(atlas2pred(input_obj, atlas_obj, atlas_mask_obj)).astype(bool)
    pred_bbox     = mask2bbox(pred_mask_arr)
    
    # metric
    align0_dice = compute_dice_coefficient(gt_mask_arr, pred_mask_arr)
    
    print("Starting csc")
    
    # cascade: expand margin around pred_bbox
    imin, imax, jmin, jmax, kmin, kmax = pred_bbox
    imin_pad, imax_pad, jmin_pad, jmax_pad, kmin_pad, kmax_pad = pad_amts
    
    imin, jmin, kmin = [max(0, x-pad) for x,pad in zip((imin, jmin, kmin), (imin_pad, jmin_pad, kmin_pad))]
    imax, jmax, kmax = [min(x-pad, shape) for x,pad,shape in zip((imin, jmin, kmin),                                                                  (imin_pad, jmin_pad, kmin_pad),                                                                  input_obj.GetSize())]
                                                   
    # cascade: re-align sub-brain (ROI + margin) region
    csc = True
    try:
        csc_gt_mask_arr   = sitk2np(input_mask_obj[imin:imax, jmin:jmax, kmin:kmax]).astype(bool)   
        csc_pred_mask_arr = sitk2np(atlas2pred(input_obj[imin:imax, jmin:jmax, kmin:kmax],                                                 atlas_obj[imin:imax, jmin:jmax, kmin:kmax],                                                 atlas_mask_obj[imin:imax, jmin:jmax, kmin:kmax])).astype(bool)
        
        print("End csc")
        
        print("mask2bbox")
        csc_pred_bbox   = mask2bbox(pred_mask_arr)
        csc_gt_bbox     = mask2bbox(csc_gt_mask_arr)
        
        # metric
        align1_dice = compute_dice_coefficient(csc_gt_mask_arr, csc_pred_mask_arr)
        
    except:
        print("Error")
        csc = False
        
        csc_pred_bbox = pred_bbox
        csc_gt_bbox   = gt_bbox
        align1_dice   = align0_dice
        
        # store error
        if (input_idx, atlas_idx) in csc_errors:
            csc_errors[(input_idx, atlas_idx)].append(pad_dict)
        else:
            csc_errors[(input_idx, atlas_idx)] = [pad_dict]

   
    print("Return row")
    
    # store input idx, atlas idx, pad amts, bbox delta between gt and pred 
    return {"input_idx": input_idx, "atlas_idx": atlas_idx,             "align0": align0, "align1": align1, "csc": csc,             **pad_dict, **bbox2diff(csc_gt_bbox, csc_pred_bbox)}


# In[33]:


pad_ranges = (imin_range, imax_range, jmin_range, jmax_range, kmin_range, kmax_range)
for r in pad_ranges:
    print(f"Len {len(r):2}: ", r)

from functools import reduce
product = reduce((lambda x, y: x * y), [len(r) for r in pad_ranges])
print(f"Total No. Parameter Combinations: {product}.")


# In[34]:


params = [(imin_pad, imax_pad, jmin_pad, jmax_pad, kmin_pad, kmax_pad)           for imin_pad in imin_range           for imax_pad in imax_range           for jmin_pad in jmin_range           for jmax_pad in jmax_range           for kmin_pad in kmin_range           for kmax_pad in kmax_range
         ]


# In[35]:


print(f"Len params {len(params)}")
print(*params[:10], sep="\n")


# In[37]:


# part 0

input_idx, atlas_idx = idxs[0]
pad_amts = params[0]

print("INPUTS: ", input_idx, atlas_idx, pad_amts)


# In[ ]:


start = time.time()
test = idxs2csc_bbox(input_idx, atlas_idx, pad_amts)
elapsed = time.time() - start
print(f"Elapsed {elapsed}")

