#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Save N4 bias field correction

# # Imports

# In[36]:


import os

# Paths to (1) code (2) data (3) saved models
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit/PitMRdata"
model_src   = "/gpfs/data/oermannlab/private_data/DeepPit/saved_models"

# save_src = "/gpfs/data/oermannlab/private_data/DeepPit/saved_landmarks/ABIDE"
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


# In[37]:


# imports

# Utilities
import os
import sys
import time
import glob
import pickle
from pathlib import Path
# sys.path.append('/gpfs/home/gologr01/DeepPit')
# sys.path.append('/gpfs/home/gologr01/OBELISK')

# Numpy torch pandas
import torch

# imports
import SimpleITK as sitk
import meshio
from helpers.preprocess import seg2mask, get_data_dict


# # MR data

# In[38]:


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


# # Get chunk

# In[41]:


import os
taskid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
   
n_total = len(fnames_no_pad)

chunk_len = 20    
chunks    = [range(i,min(i+chunk_len, n_total)) for i in range(0, n_total, chunk_len)]

print(f"N_chunks = {len(chunks)}")
# print(f"Array Task ID: {taskid}")
# print(f"Array ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
# print(f"Job ID: {os.getenv('SLURM_JOB_ID')}")
#print(*chunks, sep="\n")

task_chunk = chunks[taskid]


# In[23]:


def is_todo(f):
    children = os.listdir(f)
    return len(children) == 1
    
files     = [fnames_no_pad[i] for i in task_chunk]
nii_files = [os.path.join(f, os.listdir(f)[0]) for f in files if is_todo(f)]


# # Process

# In[8]:


# from FAIMED3D 02_preprocessing
# and https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
print("tot ", len(nii_files))

count = 0
for mr_path in nii_files: 
    # print
    print(count, mr_path, flush=True)
    count += 1
    
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


# In[ ]:




