#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Save reorient

# # Imports

# In[1]:


# NYU
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit"


# In[2]:


# UMich 
# code src: "/home/labcomputer/Desktop/Rachel"
# data src: "../../../../..//media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/rachel/"


# In[3]:


import os

# Paths to (1) code (2) data (3) saved models (4) saved metadata
deepPit_src = f"{code_src}/DeepPit"
obelisk_src = f"{code_src}/OBELISK"

# saved models, dset metadata
model_src  = f"{data_src}/saved_models"
dsetmd_src = f"{data_src}/saved_dset_metadata"

# dsets
dsets_src    = f"{data_src}/PitMRdata"

# key,val = dset_name, path to top level dir
dset_dict = {
    "ABIDE"                  : f"{dsets_src}/ABIDE",
    "ABVIB"                  : f"{dsets_src}/ABVIB/ABVIB",
    "ADNI1_Complete_1Yr_1.5T": f"{dsets_src}/ADNI/ADNI1_Complete_1Yr_1.5T/ADNI",
    "AIBL"                   : f"{dsets_src}/AIBL/AIBL",
    "ICMB"                   : f"{dsets_src}/ICMB/ICBM",
    "PPMI"                   : f"{dsets_src}/PPMI/PPMI",
}

# print
print("Folders in dset src: ", end=""); print(*os.listdir(dsets_src), sep=", ")


# In[4]:


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

# In[5]:


#dset_name = "PPMI" #"AIBL" #"ABVIB" #"ICMB" # "AIBL" # "ABVIB" #PPMI
#dset_src  = dset_dict[dset_name]

all_fnames = []

for dset_name in dset_dict.keys():
    with open(f"{dsetmd_src}/{dset_name}_fnames.txt", "rb") as f:
        all_fnames += pickle.load(f)

fnames = all_fnames
# load files
# with open(f"{dsetmd_src}/{dset_name}_dset_metadata.pkl", "rb") as f:
#     q_t1 = pickle.load(f)

# with open(f"{dsetmd_src}/{dset_name}_fnames.txt", "rb") as f:
#     fnames = pickle.load(f)


# In[6]:


print(dset_name, len(fnames)) #, *fnames, sep="\n")


# # Test/Cleanup

# In[7]:


corrected = []
uncorrected = []
multiple    = []

def is_corrected(f):
    nii_paths = glob.glob(f"{f}/*corrected_n4.nii")
    
    # filter las_corrected_n4.nii
    nii_paths = [n for n in nii_paths if not n.endswith("las_corrected_n4.nii")]
    
    if len(nii_paths) == 1: 
        corrected.append(f)
        return True
    
    if len(nii_paths) == 0: 
        uncorrected.append(f)
        return False
    
    if len(nii_paths) > 1: 
        multiple.append(f)
        return True  
                
for f in fnames:
    is_corrected(f)
    
print(f"Corrected: {len(corrected)}, TODO: {len(uncorrected)}, Dupl: {len(multiple)}")


# In[15]:


las_corrected = []
las_uncorrected = []
las_multiple    = []

def is_las_corrected(f):
    nii_paths = glob.glob(f"{f}/*las_corrected_n4.nii")
    
    if len(nii_paths) == 1: 
        las_corrected.append(f)
        return True
    
    if len(nii_paths) == 0: 
        las_uncorrected.append(f)
        return False
    
    if len(nii_paths) > 1: 
        las_multiple.append(f)
        return True  
                
for f in fnames:
    is_las_corrected(f)
    
print(f"LAS: Corrected: {len(las_corrected)}, TODO: {len(las_uncorrected)}, Dupl: {len(las_multiple)}")


# In[16]:


las_uncorrected = [f for f in las_uncorrected if f in corrected]
print("las TODO: ", len(las_uncorrected))


# # Get chunk

# In[17]:


import os

try:
    taskid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    taskid = 0
    
n_total = len(las_uncorrected)

chunk_len = 5    
chunks    = [range(i,min(i+chunk_len, n_total)) for i in range(0, n_total, chunk_len)]

print(f"N_chunks = {len(chunks)}")
# print(f"Array Task ID: {taskid}")
# print(f"Array ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
# print(f"Job ID: {os.getenv('SLURM_JOB_ID')}")
#print(*chunks, sep="\n")

task_chunk = chunks[taskid]


# In[11]:


def read_corrected_n4(fn):
    #im_path = f"{fn}/corrected_n4.nii"
    nii_paths = glob.glob(f"{f}/*corrected_n4.nii")
    im_path   = nii_paths[0] 
    return sitk.ReadImage(im_path, sitk.sitkFloat32)

def read_dcm(fn):
    im = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(fn), sitk.sitkFloat32)
    return im

def read_nii(fn):
    im = sitk.ReadImage(fn, sitk.sitkFloat32)    
    return im

# dcm
    #reader = sitk.ImageSeriesReader()
    #dicom_names = reader.GetGDCMSeriesFileNames(fn)
    #reader.SetFileNames(dicom_names)
    #im = reader.Execute() 
    


# # Process

# In[12]:


uncorrected_chunk = [las_uncorrected[i] for i in task_chunk]
print(len(uncorrected_chunk), *uncorrected_chunk, sep="\n")


# In[ ]:


# from FAIMED3D 02_preprocessing
# and https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html

count = 0
for mr_path in uncorrected_chunk: 
    
    start = time.time()
    
    # print
    print(count, mr_path, flush=True)
    count += 1
    
    # Read in image
    try:
        inputImage = read_corrected_n4(mr_path)
        
        # correct
        corrected_image = sitk.DICOMOrient(inputImage, "LAS")

        # write image
        corrected_fn = f"{mr_path}/las_corrected_n4.nii"
        sitk.WriteImage(corrected_image, corrected_fn)

        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:0.2f} s")
    
    except Exception as e:
        print("Skipped: ", mr_path)
        print(e)
        #inputImage = read_dcm(mr_path) 
    


# In[ ]:


print("Done.")


# In[ ]:





# In[ ]:


#print("Uncorrected: ", *uncorrected, sep="\n")

#os.listdir(uncorrected[0])
#uncorrected_nii = [os.path.join(f, os.listdir(f)[0]) for f in uncorrected]
#uncorrected_nii

# Very strange
# os.remove('/gpfs/data/oermannlab/private_data/DeepPit/PitMRdata/ABIDE/ABIDE/50455/MP-RAGE/2000-01-01_00_00_00.0/S165455/._ABIDE_50455_MRI_MP-RAGE_br_raw_20120831000745302_S165455_I329465.nii')

# # delete multiple
# for f in multiple:
#     nii_paths = glob.glob(f"{f}/*corrected_n4_corrected_n4.nii")
#     for p in nii_paths:
#         os.remove(p)

# process uncorrected
# from FAIMED3D 02_preprocessing
# and https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html

# print("tot ", len(uncorrected))

# uncorrected_nii = [os.path.join(f, os.listdir(f)[0]) for f in uncorrected]

# count = 0
# for mr_path in uncorrected_nii: 
#     # print
#     print(count, mr_path, flush=True)
#     count += 1
    
#     # Read in image
#     inputImage = sitk.ReadImage(mr_path, sitk.sitkFloat32)
    
#     # Mask the head to estimate bias
#     maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    
#     # Set corrector
#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     corrector.SetMaximumNumberOfIterations([3] * 3)
#     corrected_image = corrector.Execute(inputImage, maskImage)

#     # write image
#     corrected_fn = mr_path[:-4] + "_corrected_n4.nii"
#     sitk.WriteImage(corrected_image, corrected_fn)

