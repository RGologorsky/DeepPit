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

# In[99]:


# MODEL
model_type = "UNETR" #"SegResNetVAE" # "UNET3D" #"OBELISKHYBRID" #"VNET" # "UNET3D" "OBELISKHYBRID"
loss_type  = "log_cosh_dice_loss" #"vae_loss" #"perim_loss" #"log_cosh_dice_loss" # DICE

# DATALOADER PARAMS
bs          = 1
nepochs     = 60
num_workers = 2

# PREPROCESS (Isotropic, PadResize)
# iso_sz    = 2
# maxs      = [144, 144, 144]

iso_sz    = 3
maxs      = [96, 96, 96]

# Train:Valid:Test = 60:20:20
valid_frac, test_frac = .20, .20


# # Setup paths

# In[100]:


import os

# Paths to (1) code (2) data
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit"

# stored code
deepPit_src = f"{code_src}/DeepPit"
obelisk_src = f"{code_src}/OBELISK"

# stored data
model_src   = f"{data_src}/saved_models"
label_src   = f"{data_src}/PitMRdata/samir_labels"
ABIDE_src   = f"{data_src}/PitMRdata/ABIDE"

# stored runs Tensorboard
run_src     = f"{data_src}/runs"

# print
print("Folders in data src: ", end=""); print(*os.listdir(data_src), sep=", ")
print("Folders in label src (data w labels): ", end=""); print(*os.listdir(label_src), sep=", ")
print("Folders in ABIDE src (data wo labels) ", end=""); print(*os.listdir(ABIDE_src), sep=", ")


# # Imports

# In[101]:


# %load_ext autoreload
# %autoreload 2


# In[102]:


# imports (# Piece)
from transforms import AddChannel, Iso, PadSz,                       ZScale,                        GNoise, GBlur,                       RandBright, RandContrast,                        RandDihedral, MattAff
        
        
from helpers.losses import dice_score, dice_loss, dice_ce_loss, log_cosh_dice_loss, perim_loss

# MONAI
from monai.losses        import DiceLoss
from monai.metrics       import DiceMetric
from monai.networks.nets import VNet, UNet, SegResNetVAE, HighResNet, UNETR

# Utilities
import os, sys, gc, time, pickle
from pathlib import Path

# Input IO
import SimpleITK as sitk
import meshio

# Numpy and Pandas
import numpy as np
import pandas as pd
from pandas import DataFrame as DF

# Fastai + distributed training
from fastai              import *
from fastai.torch_basics import *
from fastai.basics       import *
from fastai.distributed  import *
from fastai.callback.all import SaveModelCallback, CSVLogger
from fastai.callback.tensorboard import TensorBoardCallback

# PyTorch
from torch import nn

# Obelisk
sys.path.append(deepPit_src)
sys.path.append(obelisk_src)

# OBELISK
from utils  import *
from models import obelisk_visceral, obeliskhybrid_visceral

# Helper functions
from helpers.preprocess import get_data_dict_n4, mask2bbox, print_bbox, get_bbox_size, print_bbox_size
from helpers.general    import sitk2np, np2sitk, print_sitk_info, lrange, lmap, numbers2groups, print_hardware_stats
from helpers.viz        import viz_axis, viz_compare_inputs, viz_compare_outputs
from helpers.time       import time_one_batch, get_time_id


# # Data

# In[103]:


folders = sorted(Path(label_src).iterdir(), key=os.path.getmtime, reverse=True)
# print(*[Path(f).name for f in folders], sep="\n")

cross_lbl_folders = folders[:5]
abide_lbl_folders = folders[5:]

print("Cross", *cross_lbl_folders, sep="\n"); print("*"*50)
print("Abide", *abide_lbl_folders, sep="\n")


# In[104]:


# Get data dict
data = {}
folders = abide_lbl_folders
for folder in folders: data.update(get_data_dict_n4(folder))

# Convert data dict => items (path to MR, path to Segm tensor)
items = list(data.values())
print(f"Full: {len(items)}")


# In[105]:


# remove bad label 50132
weird_lbls = [50132, 50403]
def is_weird(fn): return any([str(lbl) in fn for lbl in weird_lbls])
   
items = [o for o in items if not is_weird(o[0])]
print(f"Removed {len(weird_lbls)} weird, new total: {len(items)}")


# # Split train/valid/test split

# In[106]:


# save test set indices
with open(f"{data_src}/saved_dset_metadata/split_train_valid_test.pkl", 'rb') as f:
    train_idxs, valid_idxs, test_idxs, train_items, valid_items, test_items = pickle.load(f)
    tr_len, va_len, te_len =  len(train_items), len(valid_items), len(test_items)
    print("train, valid, test", tr_len, va_len, te_len, "total", tr_len + va_len + te_len)


# In[107]:


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


# In[108]:


# # save test set indices
# with open(f"{data_src}/saved_dset_metadata/split_train_valid_test.pkl", 'wb') as f:
#     pickle.dump([train_idxs, valid_idxs, test_idxs, train_items, valid_items, test_items], f)


# # Transforms

# In[109]:


p = 0.8

item_tfms  = [Iso(iso_sz), PadSz(maxs)]
batch_tfms = [ZScale(), MattAff(p=p, strength=0.05), AddChannel()]

# batch_tfms = [
#     # normalize mean/std of foreground pixels
#     ZScale(),
#     # flip
#     RandDihedral(p=p),
#     # noise
#     GNoise(p=p, std_range=[0.01, 0.1]),
#     #GBlur(p=p,  kernel_size_range=[5, 11], sigma=0.5),
#     AddChannel(),
#     # affine
#     MattAff(p=p, strength=0.05)
# ]


# # Dataloaders

# In[110]:


# tls, dls, cuda
tls = TfmdLists(items, item_tfms, splits=(train_idxs, valid_idxs))
dls = tls.dataloaders(bs=bs, after_batch=batch_tfms, num_workers=num_workers)
dls = dls.cuda()


# In[111]:


# test get one batch
# time_one_batch(dls)


# # Model

# In[112]:


full_res = maxs
print(f"Full res: {full_res}")


# In[113]:


if model_type == "VNET":
    # https://docs.monai.io/en/latest/networks.html#vnet
    device = torch.device("cuda:0")
    model = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    ).to(device)
    
elif model_type == "UNET3D":
    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
elif model_type == "UNETR":
    device = torch.device("cuda:0")
    model = UNETR(
        in_channels=1, 
        out_channels=2, 
        img_size=full_res, 
        feature_size=16, 
        hidden_size=768, 
        mlp_dim=3072, 
        num_heads=12, 
        pos_embed='perceptron', 
        norm_name='instance', 
        conv_block=False, 
        res_block=True, 
        dropout_rate=0.0
    ).to(device)

elif model_type == "SegResNetVAE":
    device = torch.device("cuda:0")
    model = SegResNetVAE(
        input_image_size = full_res, 
        vae_estimate_std=False, 
        vae_default_std=0.3, 
        vae_nz=256, 
        spatial_dims=3, 
        init_filters=8, 
        in_channels=1, 
        out_channels=2, 
        dropout_prob=None, 
        act=('RELU', {'inplace': True}), 
        norm=('GROUP', {'num_groups': 8}), 
        use_conv_final=True, 
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), #upsample_mode=<UpsampleMode.NONTRAINABLE: 'nontrainable'>
    ).to(device)
    
elif model_type == "OBELISKHYBRID":
    full_res = maxs
    model    = obeliskhybrid_visceral(num_labels=2, full_res=full_res)


# # Loss

# In[ ]:


# def vae_loss(preds, targets):
#     tens_preds =
#     dice_loss = log_cosh_dice_loss(preds[0])
#     vae_loss_l2 = (preds-targets)


# In[ ]:


if loss_type == "log_cosh_dice_loss":
    loss_function = log_cosh_dice_loss
elif loss_type == "DICE":
    loss_function = dice_loss
    
elif loss_type == "perim_loss":
    loss_function = log_cosh_dice_loss
    #loss_function = perim_loss
    
elif loss_type == "vae_loss":
    loss_function = vae_loss
else:
    loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=False)


# In[ ]:


print(loss_type)


# In[ ]:


# Save test idxs + model + runs

# file name
model_time = rank0_first(lambda:get_time_id()) # 'Mon Oct 18 13:35:29 2010'
model_name = f"model_{model_type}_loss_{loss_type}_iso_{iso_sz}mm_pad_{maxs[0]}_{maxs[1]}_{maxs[2]}_bs_{bs}_epochs_{nepochs}_time_{model_time}"
print(f"Model name: {model_name}")


# # Save

# In[114]:


# make dir
model_src = f"{run_src}/{model_name}"
fig_src = f"{model_src}/figs"
Path(fig_src).mkdir(parents=True, exist_ok=True)


# In[115]:


# save test set indices
with open(f"{model_src}/test_items.pkl", 'wb') as f:
    rank0_first(lambda:pickle.dump(list(test_items), f))
   
# save data augs
with open(f"{model_src}/data_augs.txt", 'w') as f:
    s = "\n".join([str(tfm) for tfm in item_tfms + batch_tfms])
    rank0_first(lambda: print(s, file=f))
    
# with open(f"{model_src}/{model_name}_test_items.pkl", 'wb') as f:
#     rank0_first(lambda:pickle.dump(list(test_items), f))


# # Callbacks

# In[116]:


class PerimLossMidway(Callback):
    def before_epoch(self): 
        if self.epoch == self.n_epoch//2:
            self.learn.loss_func = perim_loss
            print("Changed ", self.learn.loss_func, "at epoch ", self.n_epoch//2)
            
cbs = [SaveModelCallback(monitor='dice_score', with_opt=True), CSVLogger(fname=f"{fig_src}/history.csv")]

if loss_type == "perim_loss":
    print("added PerimLoss callback")
    cbs.append(PerimLossMidway())


# # Learner

# In[117]:


# clear cache
gc.collect()
torch.cuda.empty_cache()
print_hardware_stats()


# In[118]:


learn = rank0_first(lambda:
            Learner(dls   = dls, \
                model     = model, \
                loss_func = loss_function, \
                metrics   = dice_score, \
                model_dir = model_src, \
                cbs       = cbs)
        )

# cbs TensorBoardCallback(Path(run_src)/model_name, trace_model=True)
# GPU
learn.model = rank0_first(lambda:learn.model.cuda())


# In[119]:


# check
print("Check")
b = dls.one_batch()
xb,yb = b
print(f"Batch: {len(b)}. xb: {xb.shape}, yb: {yb.shape}")
predb = learn.model(xb)
print(f"Pred batch: {predb.shape}")
loss = loss_function(predb, yb)
print(f"Loss: {loss}")


# In[ ]:


# print(predb[0].shape, predb[1])


# # LR Finder

# In[93]:


# print("PRE learn.fit one cycle")
# with learn.distrib_ctx():
#     learn.fit_one_cycle(2, 3e-3, wd = 1e-4)


# In[94]:


# learn.lr_find()


# In[95]:


print("PRE learn.fit one cycle")

with learn.distrib_ctx():
    learn.fit_one_cycle(nepochs, 3e-3, wd = 1e-4)
    


# In[ ]:


#     if loss_type == "perim_loss":
#         learn.loss_func = perim_loss
#         print("switched loss at epoch ", nepochs//2)
    
#     learn.fit_one_cycle(nepochs//2, 3e-3, wd = 1e-4)


# In[ ]:


# learn.recorder.plot_loss()


# In[95]:


def save_plot_loss(self, skip_start=5, with_valid=True):
        plt.plot(list(range(skip_start, len(self.losses))), self.losses[skip_start:], label='train')
        if with_valid:
            idx = (np.array(self.iters)<skip_start).sum()
            plt.plot(self.iters[idx:], L(self.values[idx:]).itemgot(1), label='valid')
            plt.legend()
        plt.savefig(f'{fig_src}/loss.png', bbox_inches='tight')
        plt.close()


# In[96]:


# save_plot_loss(learn.recorder)


# In[97]:


@delegates(subplots)
def save_plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    #plt.show()
    plt.savefig(f'{fig_src}/metrics.png', bbox_inches='tight')
    plt.close()


# In[98]:


save_plot_metrics(learn.recorder)


# In[105]:


c = learn.cbs[-2]
get_ipython().run_line_magic('pinfo2', 'c')


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

