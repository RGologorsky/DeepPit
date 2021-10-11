#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Save re-oriented to LAS, isotropic.

# In[107]:


new_sp = 2


# In[108]:


import os

try:
    taskid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
except:
    taskid = 0
    
print(f"Taskid: {taskid}")


# In[109]:


# print(f"Array Task ID: {taskid}")
# print(f"Array ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
# print(f"Job ID: {os.getenv('SLURM_JOB_ID')}")
#print(*chunks, sep="\n"


# # Imports

# In[110]:


# imports

# Utilities
import os, sys, time, glob, pickle
from pathlib import Path

# Numpy torch pandas
import torch
import torch.nn.functional as F

import meshio
import SimpleITK as sitk
from helpers.preprocess import seg2mask, get_data_dict


# In[111]:


# NYU
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit"


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


# In[112]:


# UMich 
# code src: "/home/labcomputer/Desktop/Rachel"
# data src: "../../../../..//media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/rachel/"


# # Load data

# In[113]:


# load filenames
fnames = []
for dset_name in dset_dict.keys(): # # ("AIBL", "ABVIB", "ICMB", "PPMI"):
    dset_src  = dset_dict[dset_name]
    with open(f"{dsetmd_src}/{dset_name}_fnames.txt", "rb") as f:
        fnames += pickle.load(f)
        
# filter ._ prefix
fnames = [f for f in fnames if not f.startswith("._")]
print("Total n: ", len(fnames))


# In[114]:


start = time.time()

matches = [len(glob.glob(f"{f}/*las.pt")) for f in fnames]


todo = [f for i,f in enumerate(fnames) if matches[i]==0]
done = [f for i,f in enumerate(fnames) if matches[i]==1]
mult = [f for i,f in enumerate(fnames) if matches[i]>1]

print(f"Corrected: {len(done)}, TODO: {len(todo)}, Dupl: {len(mult)}")

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} s")


# In[96]:


# examples[fil[0][0]] suffixes: las_corrected_n4, corrected_n4 .nii


# # Get chunk

# In[86]:


n_total = len(todo)

chunk_len = 50    
chunks    = [range(i,min(i+chunk_len, n_total)) for i in range(0, n_total, chunk_len)]

print(f"N_chunks = {len(chunks)}")

task_chunk = chunks[taskid]


# In[87]:


todo_chunk = [todo[i] for i in task_chunk]
print(f"Todo chunk ({len(todo_chunk)}/{len(todo)})", *todo_chunk[:2], sep="\n")


# # Process

# In[43]:


# fn is path to terminal folder
def read_dcm(fn):
    dcms = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(fn)
    if len(dcms) == 1: dcms = dcms[0]   
    im = sitk.ReadImage(dcms, sitk.sitkFloat32)
    return im

def read_nii(fn):
    niis = [f for f in os.listdir(fn) if f.endswith(".nii") and not f.endswith("corrected_n4.nii") and not f.startswith("._")]
    nii   = niis[0]
    im = sitk.ReadImage(f"{fn}/{nii}", sitk.sitkFloat32)    
    return im


# In[60]:


def process(inputImage):
    # 1. Reorient, mr to im tensor
    mr = sitk.DICOMOrient(inputImage, "LAS")
    im = torch.transpose(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)

    # 2. Resize so isotropic spacing
    orig_sp = mr.GetSpacing()
    orig_sz = mr.GetSize()
    new_sz = [int(round(osz*ospc/new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

    while im.ndim < 5: 
        im = im.unsqueeze(0)
        
    return F.interpolate(im, size = new_sz, mode = 'trilinear', align_corners=False).squeeze()


# In[67]:


# # test
# mr_path = todo[i]
# try:
#     inputImage = read_nii(mr_path)
# except:
#     inputImage = read_dcm(mr_path) 
            
# z = process(inputImage)


# In[97]:


# from FAIMED3D 02_preprocessing
# and https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html

skipped = 0
count = 0
for mr_path in todo_chunk: 
    
    start = time.time()
    
    try:
        
        # Print
        print(count, mr_path)
        count += 1

        # Read in image
        try:
            inputImage = read_nii(mr_path)
        except:
            inputImage = read_dcm(mr_path) 

        # correct
        corrected_image = process(inputImage)
    
        # write image
        corrected_fn = f"{mr_path}/iso_{new_sp}_las.pt"
        torch.save(corrected_image, corrected_fn)
    
    except Exception as e:
        print("Skipped: ", mr_path)
        print(e)
        skipped += 1
    
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:0.2f} s")
    
print(f"Skipped: {skipped}")


# In[70]:


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

