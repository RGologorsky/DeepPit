#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Save N4 bias field correction

# # Imports

# In[1]:


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


# In[2]:


# imports

import os
import sys
import time
import glob
# sys.path.append('/gpfs/home/gologr01/DeepPit')
# sys.path.append('/gpfs/home/gologr01/OBELISK')

# imports
import SimpleITK as sitk
import torch
import meshio
from pathlib import Path
from helpers.preprocess import seg2mask, get_data_dict


# # MR data

# In[3]:


def get_data_dict_n4(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
        segm_obj_path = os.path.join(train_path, folder, "seg.pt")

        mp_path      = os.path.join(train_path, folder, "MP-RAGE")
        folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
        folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])

        # choose corrected_n4 if available
        nii_paths = glob.glob(f"{folder2_path}/*.nii")
        nii_path = nii_paths[0]
         
        if len(nii_paths) > 1 and not nii_path.endswith("corrected_n4.nii"):
            nii_path = nii_paths[1]
            
        if len(nii_paths) > 2:
            print(folder2_path)
            
        train_data_dict[folder] = (nii_path, segm_obj_path) #(segm_obj_path, nii_path)
    return train_data_dict


# In[4]:


def get_data_dict_no_n4(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
        segm_obj_path = os.path.join(train_path, folder, "seg.pt")

        mp_path      = os.path.join(train_path, folder, "MP-RAGE")
        folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
        folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])

        # choose NOT corrected_n4
        nii_paths = glob.glob(f"{folder2_path}/*.nii")
        
        # get original .nii
        for nii in nii_paths:
            if not nii.endswith("corrected_n4.nii"):
                nii_path = nii
            
        train_data_dict[folder] = (nii_path, segm_obj_path) #(segm_obj_path, nii_path)
    return train_data_dict


# In[5]:


# Get data dict
data = {}
folders = os.listdir(label_src)
for folder in folders: 
    data.update(get_data_dict_no_n4(f"{label_src}/{folder}"))

# Convert data dict => items (path to MR, path to Segm tensor)
items = list(data.values())

# filter no corrected
items_no_n4 = [item for item in items if not item[0].endswith("corrected_n4.nii")]


# In[6]:


print(len(items), len(items_no_n4), len(items)-len(items_no_n4))


# In[7]:


print(items[0][0])
print(items[0][0][:-4] + "_corrected_n4.nii")


# # Process

# In[8]:


# from FAIMED3D 02_preprocessing
# and https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html

# 45-50 secs each => 5 hrs
start = time.time()

print(f"N left: ", len(items_no_n4))

count = 0

for mr_path, seg_path in items_no_n4: 
    
    start1 = time.time()
    
    # Read in image
    inputImage = sitk.ReadImage(mr_path, sitk.sitkFloat32)
    
    # Mask the head to estimate bias
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    
    # Set corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([3] * 3)
    corrected_image = corrector.Execute(inputImage, maskImage)

    # write image
    corrected_fn = mr_path[:-4] + "_corrected_n4.nii"
    sitk.WriteImage(corrected_image, corrected_fn)

    elapsed1 = time.time() - start1
    print(f"Elapsed {elapsed1} s")

    print(f"Index {count}, fn {corrected_fn}")
    count += 1

elapsed = time.time() - start
print(f"Total Elapsed {elapsed} s")


# In[ ]:




