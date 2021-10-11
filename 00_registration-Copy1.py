#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# sitk registration. 1) rigid 2) nonrigid 3) label vote 4) save row of df in txt file w/ moving fn, atlas fn, dice, hausdorff, before and after

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


import os

try:
    taskid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    taskid = 0
    
print(f"Taskid: {taskid}")


# In[3]:


import time
import torch
import SimpleITK as sitk
import numpy as np
import pandas as pd

from helpers.preprocess import torch_mask2bbox


# # get items

# In[4]:


from helpers.items_constants import *

save_loc = f"{data_src}/saved_rigid_reg/taskid_{taskid}.pkl"
print(save_loc)


# In[5]:


print(f"ABIDE: {len(abide_lbl_items)}")
print(f"Others: {len(cross_lbl_items)}")


# In[6]:


all_lbl_items = abide_lbl_items + cross_lbl_items
print(f"All lbl items: {len(all_lbl_items)}")


# In[9]:


n_total   = len(all_lbl_items)
chunk_len = 3    
chunks    = [(i,min(i+chunk_len, n_total)) for i in range(0, n_total, chunk_len)]

print(f"N_chunks = {len(chunks)}")
# print(f"Array Task ID: {taskid}")
# print(f"Array ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
# print(f"Job ID: {os.getenv('SLURM_JOB_ID')}")
#print(*chunks, sep="\n")

task_chunk = chunks[taskid]
print(task_chunk)


# In[8]:


n_atlas = 10

# set moving fns, non-overlapped fixed image fns
# CHECK ABIDE ALL DIFFERENT STEM FOLDER
moving_image_start_index, moving_image_end_index = task_chunk

# atlas = fixed, not same patient as moving

# threshold moving indices to be within ABIDE
start = min(moving_image_start_index, len(abide_lbl_items))
end   = min(moving_image_end_index, len(abide_lbl_items))

fixed_image_indices  = list(range(0, start)) + list(range(end, len(abide_lbl_items)))
fixed_image_indices  = np.random.choice(fixed_image_indices, size=n_atlas, replace=False)

moving_image_items = [all_lbl_items[i]   for i in range(moving_image_start_index, moving_image_end_index)]
fixed_image_items  = [abide_lbl_items[i] for i in fixed_image_indices]


# # eval metrics

# In[9]:


# evaluate
filters = [sitk.LabelOverlapMeasuresImageFilter(), sitk.HausdorffDistanceImageFilter()]
methods = [
    [
        sitk.LabelOverlapMeasuresImageFilter.GetDiceCoefficient, 
        sitk.LabelOverlapMeasuresImageFilter.GetFalseNegativeError, 
        sitk.LabelOverlapMeasuresImageFilter.GetFalsePositiveError
    ],
    [sitk.HausdorffDistanceImageFilter.GetHausdorffDistance]
]

names = [
    ["dice", "false_neg", "false_pos"],
    ["hausdorff_dist"]
]


# # align

# In[10]:


def rigid_intramodal_registration(fixed_image, moving_image):
    
    # registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # initial transform, T: Moving -> Fixed space, used to resample F to M's domain
    
    # Set the initial moving and optimized transforms.
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    optimized_transform = sitk.Euler3DTransform()    
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)
    
    # metric
    registration_method.SetMetricAsMattesMutualInformation()
    
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    #registration_method.SetMetricFixedMask(fixed_image_mask)
    
    # multi-res
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors    = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas= [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Execute
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Always check the reason optimization terminated.
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return final_transform
    


# In[11]:


def get_isotropic(obj, mask_obj = None, new_spacing = (1,1,1), interpolator=sitk.sitkLinear):
    """ returns obj w/ 1mm isotropic voxels """
    
    original_size    = obj.GetSize()
    original_spacing = obj.GetSpacing()

    min_spacing = min(new_spacing)

    new_size = [int(round(osz*ospc/min_spacing)) for osz,ospc in zip(original_size, original_spacing)]

    def resample(o): return sitk.Resample(o, new_size, sitk.Transform(), interpolator,
                             o.GetOrigin(), new_spacing, obj.GetDirection(), 0,
                             o.GetPixelID())
    
    return resample(obj), resample(mask_obj) if mask_obj else resample(obj)  


# In[12]:


# path to sitk objs
def load_mask(mk_path, mr):
    mk = torch.load(mk_path)
    mk = sitk.GetImageFromArray(torch.transpose(mk.byte(), 0, 2))
    mk.CopyInformation(mr)
    return mk

def load_mr_mk(mr_path, mk_path):
    mr = sitk.ReadImage(mr_path, sitk.sitkFloat32)
    mk = load_mask(mk_path, mr)
    return mr, mk

def process_input(*args):
    return [load_mr_mk(*item) for item in args]


# In[13]:


# get output segmentation
def get_segm(fixed_image, fixed_mask, moving_image):
    
    # get final transform, move M onto F's domain; inverse F onto M
    tx     = rigid_intramodal_registration(fixed_image, moving_image)
    inv_tx = tx.GetInverse()

    # get output
    fixed_mask_resampled = sitk.Resample(fixed_mask, moving_image, inv_tx, sitk.sitkNearestNeighbor,
                                         0.0, #out of bounds pixel color
                                        fixed_mask.GetPixelID())
    return fixed_mask_resampled


# In[14]:


# d{"dice": x, "Hausdorff": y, "false pos": z}
def eval_measure(ground_truth, after_registration, names_todo=None):
    if isinstance(names_todo, str): names_todo = [names_todo]
        
    d = {}
    for f,method_list, name_list in zip(filters, methods, names):
        for m,n in zip(method_list, name_list):
            if not names_todo or n in names_todo:
                try:
                    f.Execute(ground_truth, after_registration)
                    val = m(f)
                except:
                    val = "-99"
                d[n] = val
    return d


# In[1]:


print("hi")


# In[15]:


rows = []
# {moving_fn, atlas1..10_fn, metrics1...10_val}

for moving_image_item in moving_image_items:
    start = time.time()
        
    # filenames
    fns_dict = {
        "moving_fn": moving_image_item[0], 
        **{f"fixed_fn{i}": fixed_image_item[0] for i,fixed_image_item in enumerate(fixed_image_items)}
    }
             
    # load inputs
    sitk_items = process_input(moving_image_item, *fixed_image_items)
        
    # iso
    sitk_items = [get_isotropic(*item) for item in sitk_items]
            
    moving_image, moving_mask = sitk_items[0]
    fixed_items               = sitk_items[1:]
        
    # get atlas votes
    atlas_masks = [get_segm(*fixed_item, moving_image) for fixed_item in fixed_items]
    
    # get dice for indiv atlas: [{dice: x}, {dice:y}] -> {dice0: x, dice1: y}
    dices = [eval_measure(moving_mask, atlas_mask, "dice") for atlas_mask in atlas_masks]
    dices_dict = {f"{k}{i}":v for i,d in enumerate(dices) for k,v in d.items()}
    
    # get majority vote
    labelForUndecidedPixels = 1
    majority_vote = sitk.LabelVoting(atlas_masks, labelForUndecidedPixels)    
    
    # compute metrics {dice: x, "Hausdorff": y, "false pos": z, "false neg"}
    metrics = eval_measure(moving_mask, majority_vote)
    metrics_dict = {f"majority_{k}":v for k,v in metrics.items()}
    
    rows.append({**fns_dict, **dices_dict, **metrics_dict})
    
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f} s for {len(fixed_image_indices)} atlases.") 


# In[32]:


# rows_df[[f"dice{i}" for i in range(n_atlas)]+["majority_dice"]]


# In[33]:


# l = rows_df[[f"fixed_fn{i}" for i in range(n_atlas)]+["moving_fn"]].values[0]
# from pathlib import Path
# for idx,i in enumerate(l): print(idx,Path(i).parent.parent.parent.parent.name)


# In[16]:


import pandas as pd

# save as dataframe
rows_df = pd.DataFrame(rows)
rows_df.to_pickle(save_loc)


# # End

# In[17]:


print("Done")

