#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# Train model
# 
# Thanks to: OBELISK, FAIMED3D, MONAI
# - https://github.com/mattiaspaul/OBELISK
# -  https://github.com/kbressem/faimed3d/blob/main/examples/3d_segmentation.md

# # Setup parameters

# In[1]:


import os,sys

# DATALOADER PARAMS
bs          = 2
nepochs     = 60
num_workers = 2

kwargs = dict(arg.split("=") for arg in sys.argv if "=" in arg)
print(kwargs)

do_reload = "model_type" not in kwargs # True #True #True #False
if do_reload:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    
    model_type = "UNETR" # VNET, UNET3D, UNETR, SegResNetVAE, OBELISKHYBRID
    loss_type  = "DICE_loss" # DICE_loss
    pixdim     = tuple(1.5 for _ in range(3))
    full_res   = tuple(96 for _ in range(3))
    do_flip    = True
    do_simple  = False
    do_test    = False # False

else:
    model_type = kwargs["model_type"]
    loss_type  = kwargs["loss_type"]
    pixdim     = kwargs["pixdim"]
    full_res   = kwargs["full_res"]
    do_flip    = kwargs["do_flip"]
    do_simple  = kwargs["do_simple"]
    do_test    = False

    # tuple
    pixdim    = tuple(float(pixdim) for _ in range(3))
    full_res  = tuple(int(full_res) for _ in range(3))
    
    # bool
    do_flip   = do_flip == "True"
    do_simple = do_simple == "True"

    print(f"Model: {model_type}")
    print(f"Loss : {loss_type}")
    print(f"Pixd : {pixdim}")
    print(f"Fullres : {full_res}")
    print(f"Do flip: {do_flip}")
    print(f"Do simple: {do_simple}")
    
if model_type == "UNETR":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# In[2]:


# Imports

# Utilities
import os, shutil, sys, gc, time, pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Monai
from monai.config    import print_config

# Fastai + distributed training
from fastai              import *
from fastai.torch_basics import *
from fastai.basics       import *
from fastai.distributed  import *
from fastai.callback.all import SaveModelCallback, CSVLogger, ProgressCallback

# Helpers
from helpers.general import print_hardware_stats
from helpers.losses  import dice_score
from helpers.items_constants    import *
from helpers.transforms_simplified         import *

# Save test idxs + model + runs
from helpers.time       import time_one_batch, get_time_id
from helpers.model_loss_choices import get_model, get_loss

print_config()

# clear cache
gc.collect()
torch.cuda.empty_cache()
print_hardware_stats()


# In[3]:


# Items as dict
train_itemsd = getd(train_items)
val_itemsd   = getd(valid_items)
print(f"train: {len(train_itemsd)}, val: {len(val_itemsd)}")

# Transforms
print(f"{model_type}, res {full_res} simple augs {do_simple} flip {do_flip} weird {not do_simple and not do_flip}")
train_tfms, val_tfms = get_train_valid_transforms(items=train_itemsd, pixdim=pixdim, full_res=full_res, 
                                                  do_flip=do_flip, do_simple=do_simple, do_condseg=(model_type=="CONDSEG"))

# tls, dls, cuda
train_dl = TfmdDL(train_itemsd, after_item=train_tfms, after_batch=[], bs=bs)
val_dl   = TfmdDL(val_itemsd,   after_item=val_tfms,   after_batch=[], bs=bs)

# tls, dls, cuda
if do_test:
    train_dl = TfmdDL(train_itemsd[:10], after_item=train_tfms, after_batch=[], bs=bs)
    val_dl   = TfmdDL(val_itemsd[:10],   after_item=val_tfms,   after_batch=[], bs=bs)
    nepochs = 2
    
dls = DataLoaders(train_dl, val_dl)
dls = dls.cuda()

# get model
model   = get_model(model_type, full_res)
loss_fn = get_loss(loss_type) 

# save model in runs/model dir + figs; mkdir

# file name
model_time = get_time_id() # 'Mon Oct 18 13:35:29 2010'
model_name = f"model_{model_type}_loss_{loss_type}_full_res_{full_res[0]}_pixdim_{pixdim[0]}_do_simple_{do_simple}_do_flip_{do_flip}_bs_{bs}_epochs_{nepochs}_time_{model_time}"
print(f"Model name: {model_name}")

model_src = f"{run_src}/{model_name}"
fig_src   = f"{model_src}/figs"
Path(fig_src).mkdir(parents=True, exist_ok=True)

# cbs
cbs = [
    Recorder(train_metrics=True), # False
    SaveModelCallback(monitor='valid_dice_score', with_opt=True), 
    CSVLogger(fname=f"{fig_src}/history.csv")
]

# learner
learn = Learner(dls   = dls,                 model     = model,                 loss_func = loss_fn,                 metrics   = dice_score,                 model_dir = model_src,                 cbs       = [])

# remove post-recorder, add new cbs, to GPU
learn.remove_cbs(learn.cbs[1:])
learn.add_cb(ProgressCallback())
learn.add_cbs(cbs)
learn.model = learn.model.cuda()

# save data augs
with open(f"{model_src}/data_augs.txt", 'w') as f:
    def print_data_augs():
        print("Train Tfms: ", file=f); print(monai_tfms2str(train_tfms), file=f)
        print("Val   Tfms: ", file=f); print(monai_tfms2str(val_tfms), file=f)
        
    print_data_augs()


# In[4]:


# # check
# print("Check")
# b = dls.one_batch()
# xb,yb = b #b["image"], b["label"]
# print(f"Batch: {len(b)}. xb: {xb.shape}, yb: {yb.shape}")
# predb = learn.model(xb)
# print(f"Pred batch: {predb.shape}")
# loss = loss_fn(predb, yb)
# print(f"Loss: {loss}")


# In[5]:


# print(predb[0].shape, predb[1])


# # LR Finder

# In[6]:


# print("PRE learn.fit one cycle")
# with learn.distrib_ctx():
#     learn.fit_one_cycle(2, 3e-3, wd = 1e-4)


# In[7]:


# SuggestedLRs(valley=tensor(0.0229))
# learn.lr_find()


# In[8]:


print("PRE learn.fit one cycle")
learn.fit_one_cycle(nepochs, 3e-3, wd = 1e-4)
    


# In[9]:


@delegates(subplots)
def save_plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    # 'train_loss', 'train_dice_score', 'valid_loss', 'valid_dice_score'
    names = self.metric_names[1:-1]
    print("Metric names: ", names)
    
    names_train = [n for n in names if n.startswith("train")]
    n = len(names_train)
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize)
    for i, ax in enumerate(axs):
        name = names_train[i]
        n = name[name.index("_")+1:]
        valid_name = f"valid_{n}"
        valid_idx = names.index(valid_name)
        print(i, valid_idx, name, valid_name)
        ax.plot(metrics[:, i], color='#1f77b4',  label='train')
        ax.plot(metrics[:, valid_idx], color = '#ff7f0e', label='valid')
        ax.set_title(n)
        ax.legend(loc='best')
    #plt.show()
    plt.savefig(f'{fig_src}/metrics.png', bbox_inches='tight')
    plt.close()


# In[10]:


save_plot_metrics(learn.recorder)


# In[11]:


print(model_name)


# In[12]:


# if not do_test:
#     sys.stdout = old_stdout
#     log_file.close()


# # Old

# In[ ]:


# batch_tfms = [
#     # normalize mean/std of foreground pixels
#     ZScale(),
#     # affine + flips
#     RandomAffine(p=0.5, degrees=35, translate=0.1, scale=0.1),
#     RandDihedral(p=0.5),
#     # lighting
#     RandBright(p=0.5),
#     RandContrast(p=0.5),
#     # noise for generalizability
#     GNoise(p=0.5),
#     GBlur(p=0.5),
#     # add channel dim
#     AddChannel()

# UMich 
# code src: "/home/labcomputer/Desktop/Rachel"
# data src: "../../../../..//media/labcomputer/e33f6fe0-5ede-4be4-b1f2-5168b7903c7a/home/rachel/"

# ]


# # Test

# In[ ]:


# print("Test")
# xb, yb = dls.one_batch()
# xb, yb = xb.cpu(), yb.cpu()

# pb = model.cpu()(xb)
# print(xb.shape, pb.shape)
# print(f"logcosh dice loss {log_cosh_dice_loss(pb,yb)}")


# In[ ]:


# # test:

# #dls.device = "cpu"

# start = time.time()

# x,y = dls.one_batch()
# #x,y = to_cpu(x), to_cpu(y)

# pred = learn.model(x)
# loss = learn.loss_func(pred, y)

# elapsed = time.time() - start

# print(f"Elapsed: {elapsed} s")
# print("Batch: x,y")
# print(type(x), x.shape, x.dtype, "\n", type(y), y.shape, y.dtype)

# print("Pred shape")
# print(type(pred), pred.shape, pred.dtype)

# print("Loss")
# print(loss)
# print(learn.loss_func)


# # Old

# In[ ]:


# # Viz

# from helpers.viz        import viz_axis, viz_compare_inputs, viz_compare_outputs
# from helpers.preprocess import mask2bbox
# from helpers.general    import lrange

# train_dl = TfmdDL(train_itemsd, after_item=train_tfms, after_batch=[], bs=3)

# b = train_dl.one_batch()

# xb,yb = b

# print("shape xb yb", xb.shape, yb.shape)

# i = 2
# mr = np.array(xb[i].squeeze().cpu())
# mk = np.array(yb[i].squeeze().cpu())

# bbox = mask2bbox(mk)

# viz_axis(np_arr = mr, \
#     bin_mask_arr   = mk,     color1 = "yellow",  alpha1=0.3, \
#     slices=lrange(*bbox[0:2]), fixed_axis=0, \
#     axis_fn = np.rot90, \
#     title   = "Axis 0", \

#     np_arr_b = mr, \
#     bin_mask_arr_b   = mk,     color1_b = "yellow",  alpha1_b=0.3, \
#     slices_b = lrange(*bbox[2:4]), fixed_axis_b=1, \
#     title_b  = "Axis 1", \

#     np_arr_c = mr, \
#     bin_mask_arr_c   = mk,     color1_c = "yellow",  alpha1_c=0.3, \
#     slices_c = lrange(*bbox[4:6]), fixed_axis_c=2, \
#     title_c = "Axis 2", \
  
# ncols = 5, hspace=0.3, fig_mult=2)


# In[ ]:


# length  = len(items)
# indices = np.arange(length)
# np.random.shuffle(indices)
# #rank0_first(lambda: np.random.shuffle(indices))

# test_split   = int(test_frac  * length)
# valid_split  = int(valid_frac * length) + test_split

# test_idxs    = indices[:test_split] 
# valid_idxs   = indices[test_split:valid_split]
# train_idxs   = indices[valid_split:]

# train_items = [items[i] for i in train_idxs]
# valid_items = [items[i] for i in valid_idxs]
# test_items  = [items[i] for i in test_idxs]

# # print
# print(f"Total  {len(items)} items in dataset.")
# print(f"Train: {len(train_items)} items.")
# print(f"Valid: {len(valid_items)} items.")
# print(f"Test:  {len(test_items)} items.")


# In[ ]:


# # save test set indices
# with open(f"{data_src}/saved_dset_metadata/split_train_valid_test.pkl", 'wb') as f:
#     pickle.dump([train_idxs, valid_idxs, test_idxs, train_items, valid_items, test_items], f)


# In[ ]:


# test get one batch
# time_one_batch(dls)


# In[ ]:


# # Viz

# from helpers.viz        import viz_axis, viz_compare_inputs, viz_compare_outputs
# from helpers.preprocess import mask2bbox
# from helpers.general    import lrange

# train_dl = TfmdDL(train_itemsd, after_item=train_tfms, after_batch=[], bs=3)

# b = train_dl.one_batch()

# xb,yb = b

# print("shape xb yb", xb.shape, yb.shape)

# i = 2
# mr = np.array(xb[i].squeeze().cpu())
# mk = np.array(yb[i].squeeze().cpu())

# bbox = mask2bbox(mk)

# viz_axis(np_arr = mr, \
#     bin_mask_arr   = mk,     color1 = "yellow",  alpha1=0.3, \
#     slices=lrange(*bbox[0:2]), fixed_axis=0, \
#     axis_fn = np.rot90, \
#     title   = "Axis 0", \

#     np_arr_b = mr, \
#     bin_mask_arr_b   = mk,     color1_b = "yellow",  alpha1_b=0.3, \
#     slices_b = lrange(*bbox[2:4]), fixed_axis_b=1, \
#     title_b  = "Axis 1", \

#     np_arr_c = mr, \
#     bin_mask_arr_c   = mk,     color1_c = "yellow",  alpha1_c=0.3, \
#     slices_c = lrange(*bbox[4:6]), fixed_axis_c=2, \
#     title_c = "Axis 2", \
  
# ncols = 5, hspace=0.3, fig_mult=2)


# In[ ]:


# length  = len(items)
# indices = np.arange(length)
# np.random.shuffle(indices)
# #rank0_first(lambda: np.random.shuffle(indices))

# test_split   = int(test_frac  * length)
# valid_split  = int(valid_frac * length) + test_split

# test_idxs    = indices[:test_split] 
# valid_idxs   = indices[test_split:valid_split]
# train_idxs   = indices[valid_split:]

# train_items = [items[i] for i in train_idxs]
# valid_items = [items[i] for i in valid_idxs]
# test_items  = [items[i] for i in test_idxs]

# # print
# print(f"Total  {len(items)} items in dataset.")
# print(f"Train: {len(train_items)} items.")
# print(f"Valid: {len(valid_items)} items.")
# print(f"Test:  {len(test_items)} items.")


# In[ ]:





# In[ ]:




