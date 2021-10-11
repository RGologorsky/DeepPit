#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# This notebook checks model generalization performance on other dsets.
# 
# **With gratitude to**:
# - https://github.com/mattiaspaul/OBELISK
# -  https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# In[1]:


import os

try:
    taskid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    do_task = True
except:
    taskid = 0
    do_task = False

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import SimpleITK as sitk
import pandas as pd
import numpy as np

import seaborn as sns

from fastai              import *
from fastai.torch_basics import *
from fastai.basics       import *
from fastai.distributed  import *

# Learner
import gc
gc.collect()

from helpers.losses                import dice_score
from helpers.items_constants       import *
from helpers.transforms_simplified import *
from helpers.general            import rm_prefix, get_param, get_param_default,rm_prefix, modelfn2dict, torch2sitk, sitk2torch
from helpers.model_loss_choices import get_model, get_loss
from helpers.preprocess         import batch_get_bbox
from helpers.postprocess        import get_largest_connected_component, eval_measure, eval_lcc


# In[3]:


# POST LCC DF DICT

# Items as dict 
from pathlib import Path
from helpers.items_constants import *

#items  = all_test_lbl_items
items = all_test_lbl_items #ppmi, icmb, adni, aibl, abvib, test_items
itemsd = getd(items)

def is_recent(model_fn):
    dates = [f"Aug_0{x}"  for x in range(3,10)]
    dates += [f"Aug_1{x}" for x in range(0,10)]
    dates += [f"Aug_2{x}" for x in range(0,10)]
    dates += [f"Aug_3{x}" for x in range(0,10)]
    dates += [f"Sep_0{x}" for x in range(0,10)]
    return any([date in str(model_fn) for date in dates])


model_fns = sorted(Path(run_src).iterdir(), key=os.path.getmtime, reverse=True)

done = [str(model_fn) 
        for model_fn in model_fns 
        if os.path.isfile(f"{str(model_fn)}/post_lcc_df.pkl") and is_recent(model_fn)
       ]
#print(*rm_prefix(done, prefix=run_src, do_sort=True), sep="\n")
print(f"DONE: {len(done)}")

post_df_dict = {}

for done_fn in done:
    model_name = Path(done_fn).name
    #print(model_name)
    model_src = f"{run_src}/{model_name}"
    check_post_df  = pd.read_pickle(f"{model_src}/post_lcc_df.pkl")
    check_pre_df   = pd.read_pickle(f"{model_src}/pre_lcc_df.pkl")
    #check_stats_df = pd.read_pickle(f"{model_src}/stats_df.pkl")
    
    #check_stats_df = check_stats_df.style.set_caption(f"{model_name}")

    post_df_dict[model_name] = check_post_df
    
    if len(check_post_df) != len(itemsd):
        print(done_fn)
        print("Len", len(check_post_df))
        print("*" * 50)
    #display(check_post_df)
    #display(check_pre_df)
    #display(check_stats_df)


# In[4]:


# POST PROCESS OBELISK 144 BACK TO 1.5 voxel spacing and center crop to 96

# convert preds => bilinear interpolation of probabilities
from monai.transforms import Compose, Spacingd, CenterSpatialCropd
convert_to_96_tfms = Compose([
    Spacingd(keys=["pred"], pixdim=(1.5,1.5,1.5), mode=("bilinear")),
    CenterSpatialCropd(keys=["pred"], roi_size=(96, 96, 96))
])

# REDO OBELISKS
obelisk_144 = [str(model_fn) 
        for model_fn in model_fns 
        if (("OBELISK" in str(model_fn) and "full_res_144_pixdim_1.0" in str(model_fn)) and \
                (os.path.isfile(f"{str(model_fn)}/preds_batch_96_bs_5.pkl")) and \
                (not os.path.isfile(f"{str(model_fn)}/to_96_preds_batch_96_bs_5.pkl")) and \
                (os.path.isfile(f"{str(model_fn)}/figs/metrics.png")) and \
                is_recent(model_fn)
            )
       ]

print(len(obelisk_144), *obelisk_144[0:5], sep="\n")


# In[5]:


done = [model_fn 
        for model_fn in obelisk_144 
        if os.path.isfile(f"{str(model_fn)}/to_96_preds_batch_96_bs_5.pkl")
       ]
print(len(done), *done[0:5], sep="\n")


# In[8]:


# doing    
model_idx  = taskid
model_fn   = obelisk_144[model_idx]

print(f"Doing: {model_fn}")


# In[9]:


# create batches
bs        = 5
batches = [itemsd[i:min(i+bs, len(itemsd))] for i in range(0,len(itemsd),bs)]
ranges  = [range(i,min(i+bs, len(itemsd))) for i in range(0,len(itemsd),bs)]

# 2min per OBELISK
start = time.time()

model_name = Path(model_fn).name
print(model_name)

for i in range(len(batches)):
    #start2 = time.time()
    
    # open pixdim 1.0, full_res 144 preds
    with open(f"{run_src}/{model_name}/preds_batch_{i}_bs_{bs}.pkl", 'rb') as handle:
        preds_144 = pickle.load(handle)

    # transform to pixdim 1.5 full_res 96 preds
    preds_96 = convert_to_96_tfms([{"pred": pred} for pred in preds_144])

    with open(f"{run_src}/{model_name}/to_96_preds_batch_{i}_bs_{bs}.pkl", 'wb') as handle:
        pickle.dump([torch.tensor(d["pred"]) for d in preds_96], handle)

    #print(preds_144[0].shape, preds_96[0]["pred"].shape, len(preds_96))
    #elapsed2 = time.time() - start2
    #print(f"Elapsed: {elapsed2:0.2f} s.")

elapsed = time.time() - start
print(f"Elapsed: {elapsed:0.2f} s.")


# In[3]:


# not all obelisk have predictions
# for i,model_fn in enumerate(obelisk_144):
#     print(i, model_fn); print("*"*50);
#     print("N files: ", len(os.listdir(model_fn))); print("*"*50);
#     #print(os.listdir(model_fn))


# In[ ]:





# In[ ]:


print("Done")


# # End

# In[ ]:




