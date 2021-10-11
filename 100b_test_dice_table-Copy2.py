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


# In[2]:


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


if not do_task:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# INFERENCE DATALOADER PARAMS
num_workers = 1

# ITEMS

from pathlib import Path
from helpers.items_constants import *
from helpers.general import rm_prefix, get_param_default, modelfn2dict

import SimpleITK as sitk
import pandas as pd

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

ppmi  = [i for i in cross_lbl_items if dset_dict["PPMI"] in i[0]]
icmb = [i for i in cross_lbl_items if "ICMB" in i[1]]
adni = [i for i in cross_lbl_items if "ADNI1_full" in i[1]]
aibl = [i for i in cross_lbl_items if "AIBL" in i[1]]
abvib = [i for i in cross_lbl_items if "ABVIB" in i[1]]

print(len(cross_lbl_items))
print(len(ppmi)+len(icmb)+len(adni)+len(aibl)+len(abvib))
print(len(all_test_lbl_items))
print(len(cross_lbl_items)+len(test_items))

# Items as dict 
from pathlib import Path
from helpers.items_constants import *

# print(f"n = {len(itemsd)}, test items = {len(test_items)}, other dsets = {len(cross_lbl_items)}")
# print(f"first item", itemsd[0])

import os
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch

# print_config()


# In[4]:


def is_recent(model_fn):
    dates = [f"Aug_0{x}"  for x in range(2,10)]
    dates += [f"Aug_1{x}" for x in range(0,10)]
    return any([date in str(model_fn) for date in dates])


# In[8]:


from helpers.model_loss_choices import get_model, get_loss

model_fns = sorted(Path(run_src).iterdir(), key=os.path.getmtime, reverse=True)
todo = [str(model_fn) 
        for model_fn in model_fns 
        if (os.path.isfile(f"{str(model_fn)}/post_lcc_df.pkl") and \
                (os.path.isfile(f"{str(model_fn)}/figs/metrics.png")) and \
                is_recent(model_fn)
            )
       ]

print("TODO: ", len(todo))
print(*rm_prefix(todo, prefix=run_src, do_sort=True), sep="\n")

for fn in todo:
    model_dict2 = modelfn2dict(fn)
    model_type2, loss_type2, full_res2, pixdim2, do_flip2, do_simple2 =         [model_dict2[k] for k in ("model_type", "loss_type", "full_res", "pixdim", "do_flip", "do_simple")]

    print(model_type2, loss_type2, "simple augs: ", do_simple2, "flip", do_flip2, "pixdim", pixdim2, "full_res", full_res2)
    
# doing    
model_idx  = taskid
model_fn   = todo[model_idx]
model_name = Path(model_fn).name

# get params
model_dict = modelfn2dict(model_fn)
model_type, loss_type, full_res, pixdim, do_flip, do_simple =         [model_dict[k] for k in ("model_type", "loss_type", "full_res", "pixdim", "do_flip", "do_simple")]

print(f"Chosen: {model_name} (idx {model_idx})")


print(f"Model: {model_type}")
print(f"Loss : {loss_type}")
print(f"Pixd : {pixdim}")
print(f"Fullres : {full_res}")
print(f"Do flip: {do_flip}")
print(f"Do simple: {do_simple}")


# In[ ]:


# clear cache
import gc
from helpers.general import print_hardware_stats

gc.collect()

if not str(device)=="cpu":
    torch.cuda.empty_cache()
    print_hardware_stats()
    


# In[ ]:


# Transforms

from helpers.transforms_simplified import *
train_itemsd = getd(train_items) # for condseg atlas choice
print(f"{model_type}, {loss_type}, res {full_res} simple augs {do_simple} flip {do_flip} weird {not do_simple and not do_flip}")
_, val_tfms = get_train_valid_transforms(items=train_itemsd, pixdim=pixdim, full_res=full_res, 
                                              do_flip=do_flip, do_simple=do_simple, do_condseg=(model_type=="CONDSEG"))
print(f"val tfms: ", *val_tfms.transforms, sep="\n")


from helpers.general            import get_param
from helpers.model_loss_choices import get_model, get_loss

model   = get_model(model_type, full_res)
loss_fn = get_loss(loss_type) 

# print
print("Model name: ", model_name)
print(f"Model type: {model_type}. Loss type: {loss_type}.")
# Dataloaders

# Fastai + distributed training
from fastai              import *
from fastai.torch_basics import *
from fastai.basics       import *
from fastai.distributed  import *

# time it - 18s for 484 items
start = time.time()

#items  = all_test_lbl_items
items = all_test_lbl_items #ppmi, icmb, adni, aibl, abvib, test_items
itemsd = getd(items)

# tls, dls, cuda
bs  = 30
tls = TfmdLists(itemsd, val_tfms)
dls = tls.dataloaders(bs=bs, after_batch=[], num_workers=num_workers, drop_last=False, shuffle=False, shuffle_train=False)

if not str(device)=="cpu":
    dls = dls.cuda()

# end timer
elapsed = time.time() - start
print(f"Elapsed time: {elapsed:.2f} s for {len(itemsd)} items")

# Learner
import gc
gc.collect()
from helpers.losses import dice_score
learn = Learner(dls       = dls, 
                model     = model, 
                loss_func = loss_fn,
                metrics   = dice_score)

# load model fname w/o .pth extension
learn.load(f"{run_src}/{model_name}/model")
if not str(device)=="cpu":
    learn.model = learn.model.cuda()


# In[6]:


from helpers.losses import dice, dice_score


# # Post-processing
# 
# 1. Largest Connect Label

# In[7]:


from helpers.postprocess import get_largest_connected_component, eval_measure, eval_lcc


# In[10]:


# create batches
bs        = 5
batches = [itemsd[i:min(i+bs, len(itemsd))] for i in range(0,len(itemsd),bs)]


# In[11]:


# set model to evaluate model
learn.model.eval()

# device = torch.device("cuda:0")

# pre & post LCC
pre_df  = []
post_df = []

start = time.time()
              
                        
# deactivate autograd engine and reduce memory usage and speed up computations
for i,batch in enumerate(batches):
#     start_small = time.time()
    
    data = Pipeline(val_tfms)(batch)
    inputs, labels = zip(*data) # [(img,lbl), (img,lbl)] => imgs, labels
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs).cpu()

    with open(f"{run_src}/{model_name}/preds_batch_{i}_bs_{bs}.pkl", 'wb') as handle:
        pickle.dump(outputs, handle)
        
    # clean up memory
    del inputs
    del labels
    del outputs
    
    gc.collect()
    
    if str(device) != "cpu":
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    # print_hardware_stats()

#     elapsed_small = time.time() - start_small
#     print(f"Elapsed: {elapsed_small:0.2f} s")

elapsed = time.time() - start
print(f"Elapsed: {elapsed:0.2f} s for {len(itemsd)} items.")


# # End

# In[ ]:


print("Done")


# In[ ]:


# import shutil
# for i,fn in enumerate(model_fns):
#     if os.path.isfile(f"{fn}/post_lcc_df.pkl"):
#         print(i,fn)
#         os.remove(f"{fn}/post_lcc_df.pkl")
#         try:
#             os.remove(f"{fn}/pre_lcc_df.pkl")
#             os.remove(f"{fn}/stats_df.pkl")
#         except:
#             "issue"


# In[ ]:


#import shutil
#print(os.path.isfile(f"{model_fns[0]}/model.pth"))
#shutil.rmtree(model_fns[0])


# In[ ]:


# import shutil
# for i,fn in enumerate(model_fns):
#     if not os.path.isfile(f"{fn}/model.pth"):
#         print(i,fn)
#         shutil.rmtree(fn)


# In[ ]:


# for model_fn in model_fns:
#     if os.path.isfile(f"{model_fn}/post_lcc_df.pkl"):
#         print(model_fn)


# In[ ]:





# # Choices

# In[ ]:


# from helpers.general            import get_param
# from helpers.model_loss_choices import get_model, get_loss

# model_fns = sorted(Path(run_src).iterdir(), key=os.path.getmtime, reverse=True)
# todo = [str(model_fn) 
#         for model_fn in model_fns 
#         if not os.path.isfile(f"{str(model_fn)}/post_lcc_df.pkl") and "Mon_Aug_02" in str(model_fn)
#        ]

# print("TODO: ", len(todo))

# # params
# def get_param_default(name, prefix, suffix, default):
#     try:
#         return get_param(name, prefix, suffix)
#     except:
#         return default

# for model_fn in todo:
#     model_name = Path(model_fn).name

#     model_type = get_param(model_name, "model_", "_loss")

#     if "loss_bs" in model_name:
#         loss_type  = get_param(model_name, "loss_", "_bs")
#     else:
#         loss_type  = get_param(model_name, "loss_", "_full_res")

#     full_res   = get_param_default(model_name, "full_res_", "_pixdim", 96)
#     pixdim     = get_param_default(model_name, "pixdim_", "_do_simple", 1.5)
#     do_simple  = get_param_default(model_name, "do_simple_", "_do_flip", False)
#     do_flip    = get_param_default(model_name, "do_flip_", "_bs", True)

#     # tuple
#     pixdim    = tuple(float(pixdim) for _ in range(3))
#     full_res  = tuple(int(full_res) for _ in range(3))

#     # bool
#     do_flip   = do_flip == "True"
#     do_simple = do_simple == "True"

#     print(f"Model Name: {model_name}")
#     print(f"Model: {model_type}")
#     print(f"Loss : {loss_type}")
#     print(f"Pixd : {pixdim}")
#     print(f"Fullres : {full_res}")
#     print(f"Do flip: {do_flip}")
#     print(f"Do simple: {do_simple}")
    
#     print("*"*50 + "\n")

