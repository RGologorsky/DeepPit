#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# This notebook explores the distribution of bounding box ROIs post-alignment. This range of values is used to inform the hyperparameter testing range (the optimal amount to pad the bounding box for cascaded alignment).

# # Imports

# In[21]:


# imports

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


# In[14]:


# import nibabel as nib
# img = nib.load(list(train_data_dict.values())[0][1])
# print(img.shape)
# print(img.get_data_dtype())
# print(img.get_data_dtype() == np.dtype(np.int16))
# print(img.header)


# In[4]:


# 10 MRs labelled by Dr. Hollon, need nii LPS=>RAS adjustment for the mask
ras_range = range(50455, 50464+1)
obj_range = lrange(50002, 50017+1)

# takes 30sec, parallel'izing doesn't help
print(f"Reading {len(obj_range)} obj from disk")

start = time.time()
objs = [folder2objs(str(i), train_data_dict, i in ras_range) for i in obj_range]
elapsed = time.time() - start

print(f"Got nii,semg objs from disk - {elapsed} s")

# parallel_objs = Parallel(n_jobs=4)(delayed(folder2objs)(str(i), train_data_dict, i in ras_range) for i in obj_range)

# crop objs
crop_objs = [threshold_based_crop(*o) for o in objs]

# resample to standard reference domain
reference_image, reference_center = get_reference_frame(crop_objs)
rs_objs = [resample2reference(*o, reference_image, reference_center) for o in crop_objs]


# In[23]:


print("MR Objs")
print_sitk_info(objs[0][0])
print_sitk_info(objs[0][1])
print();

print("Resampled (RS) Objs")
print_sitk_info(rs_objs[0][0])
print_sitk_info(rs_objs[0][1])


del objs
del crop_objs

# ### ROI info

# In[24]:


def get_info_df_row(folder, obj):
    
    img_obj, mask_obj = obj
    
    shape0, shape1, shape2             = img_obj.GetSize()
    imin, imax, jmin, jmax, kmin, kmax = mask2bbox(sitk2np(mask_obj))
    roi_size0, roi_size1, roi_size2    = imax-imin, jmax-jmin, kmax-kmin
    
    return {"folder": folder,             "shape0": shape0, "shape1": shape1, "shape2": shape2,             "roi_size0": roi_size0, "roi_size1": roi_size1, "roi_size2": roi_size2,             "imin": imin, "imax":imax,             "jmin":jmin, "jmax": jmax,             "kmin": kmin, "kmax": kmax}
    
def get_info_df(folders, objs):
    return DF([get_info_df_row(folder, obj) for folder, obj in zip(folders, objs)])


# In[25]:


roi_df = get_info_df(obj_range, rs_objs)
#roi_df


# In[26]:


bbox_cols = ["imin", "imax", "jmin" ,"jmax", "kmin", "kmax"]
#roi_df.boxplot(column=bbox_cols)


# In[27]:


#roi_df.hist(column=bbox_cols)


# In[28]:


# for col in bbox_cols:
#     col_max, col_min = roi_df[col].max(), roi_df[col].min()
#     print(f"Col {col}: Min {col_min:3} Max {col_max:3} Range {col_max-col_min:3}")


# Note:
#     
#     - imin/imax have a tight range, w/in 15 (sagital slice numbers)
#     - jmin/jmax have a broader range, about 30 (left/right if viewed sagitally)
#     - kmin/jmax have outliers! normal range w/in 10, outliers are +20 up/down

# ### Align
# 
# Align a sample of 10 pairs of images.

# In[29]:


n_objs = len(rs_objs)
n_pairs = 10

print(f"n_objs: {n_objs}")
print(f"n_pairs: {n_pairs}")


# In[30]:


idxs = [np.random.choice(n_objs, 2, replace=False) for _ in range(n_pairs)]
idxs


# In[35]:


from helpers_predict import atlas2pred, atlas2pred_bbox


# In[39]:


start = time.time()
test = [atlas2pred_bbox(rs_objs[input_idx][0], *rs_objs[atlas_idx]) for input_idx, atlas_idx in idxs[:3]]
elapsed = time.time() - start
print(f"Atlas2pred: {n_pairs} in {elapsed} s")


# In[41]:


def atlas2pred_bbox(input_obj, atlas_obj, atlas_mask_obj): 
    return mask2bbox(sitk2np(atlas2pred(input_obj, atlas_obj, atlas_mask_obj)))


# # In[ ]:


# start = time.time()
# test = [mask2bbox(sitk2np(atlas2pred_bbox(rs_objs[input_idx][0], *rs_objs[atlas_idx]))) for input_idx, atlas_idx in idxs]
# elapsed = time.time() - start
# print(elapsed)


# # In[ ]:


# start = time.time()
# parallel_test2 = Parallel(n_jobs=4)(delayed(atlas2pred_bbox)(input_obj      = rs_objs[input_idx][0],                                                 atlas_obj      = rs_objs[atlas_idx][0],                                                 atlas_mask_obj = rs_objs[atlas_idx][1])                                 for input_idx, atlas_idx in idxs[:2])
# elapsed = time.time() - start
# print(elapsed)


# # In[ ]:


# start = time.time()
# input_idx, atlas_idx = idxs[1]
# test2 = atlas2pred(rs_objs[input_idx][0], *rs_objs[atlas_idx])
# elasped = time.time() - start
# print(elapsed)


# # In[ ]:


# from helpers_predict import atlas2pred


# # #### By how much is pred1 off?

# # In[ ]:


# # test atlas0 and atlas 5
# def get_pred_bbox(input_idx, atlas_idx):
    
       
#     pred_bbox_coord = mask2bbox(sitk2np(atlas2pred(rs_objs[input_idx][0], *rs_objs[atlas_idx])))
    
#     pimin, pimax, pjmin, pjmax, pkmin, pkmax = pred_bbox_coord
#     roi_size0, roi_size1, roi_size2    = pimax-pimin, pjmax-pjmin, pkmax-pkmin
        
#     return {
#         f"input_idx": input_idx, \
#         f"atlas_idx": atlas_idx, \
#         "roi_size0": roi_size0, "roi_size1": roi_size1, "roi_size2": roi_size2, \
#         "imin": imin, "imax":imax, \
#         "jmin":jmin, "jmax": jmax, \
#         "kmin": kmin, "kmax": kmax
#     }
        
   


# # In[ ]:


# start = time.time()
# pred_bboxs = DF([get_pred_bbox(*idx_pair) for idx_pair in idxs])
# elapsed = time.time() - start
# print(elapsed)


# # In[ ]:


# pred_bbox_df = DataFrame(pred_bboxs)
# pred_bbox_df


# # In[ ]:


# def get_pred_bbox_coords(input_idx, atlas_idx):
#     return pred_bbox_df[(pred_bbox_df["input_idx"] == input_idx)&(pred_bbox_df["atlas_idx"]==atlas_idx)][["imin", "imax", "jmin", "jmax", "kmin", "kmax"]].values[0]

# def get_input_bbox_coords(input_idx):
#     return input_bbox_coords[input_idx]


# # In[ ]:


# def get_bbox_diff(pred_bbox, gt_bbox): 
#     diff = (x1-x2 for x1,x2 in zip(gt_bbox, pred_bbox))
#     cols = "delta_imin", "delta_imax", "delta_jmin", "delta_jmax", "delta_kmin", "delta_kmax"
#     return dict(zip(cols, diff))

# def get_keys(d, key_list):
#     return [d[k] for k in key_list]


# # In[ ]:


# bbox_diffs = [{"input_idx": input_idx, "atlas_idx": atlas_idx,                **get_bbox_diff(get_pred_bbox_coords(input_idx, atlas_idx), get_input_bbox_coords(input_idx))}               for input_idx in range(0,5) for atlas_idx in (0, 5)]


# # In[ ]:


# bbox_diffs = DataFrame(bbox_diffs)
# bbox_diffs


# # In[ ]:


# bbox_diffs.boxplot(column=["delta_imin", "delta_imax", "delta_jmin", "delta_jmax", "delta_kmin", "delta_kmax"])

