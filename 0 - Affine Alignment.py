#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[2]:


# imports
import os, sys

# filter filenames for .nii
import glob

# numpy
import numpy as np

# simple itk for dicom
import SimpleITK as sitk

# meshio for 3DSlicer segm obj
import meshio

# import sitk_gui

# segmentation, viz fns, misc
#from helpers import seg2mask, get_roi_range,                     sitk2np, np2sitk,                     viz_objs, viz_axis,                     round_tuple, lrange
# Helper fns

# numpy
import numpy as np

# viz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec

# segmentation
from scipy.spatial   import Delaunay

# numpy to SITK conversion
import SimpleITK as sitk

# Convert segmentation object to numpy binary mask
# 1. Get affine matrix in SITK (aff tfm: idx coord => physical space coord)
# 2. Convert image idxs to physical coords
# 3. Check whether the image physical coord is in the Delauney triangulation of segmented mesh points

# 1. Get affine matrix in SITK
# https://niftynet.readthedocs.io/en/v0.2.2/_modules/niftynet/io/simple_itk_as_nibabel.html
def make_affine(simpleITKImage):
    # get affine transform in LPS
    c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
         for p in ((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1),
                   (0, 0, 0))]
    c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine

# Seg2mask
def seg2mask(image_obj, segm_obj):
    dims = image_obj.GetSize()
    aff     = make_affine(image_obj)
    idx_pts = np.indices(dims[::-1]).T.reshape(-1,3)[:,[2,1,0]]
    physical_pts = (np.dot(aff[:3,:3], idx_pts.T) + aff[:3,3:4]).T 
    return (Delaunay(segm_obj.points).find_simplex(physical_pts) >= 0).reshape(dims)

## Viz multi-axes
# Vz fns

def get_mid_idx(vol, ax): return vol.shape[ax]//2
def get_mid_idxs(vol): return [get_mid_idx(vol, ax=i) for i in (0, 1, 2)]


# ne.plot.slices(slices, titles=titles, grid=[2,3], \
#                cmaps=['gray'], do_colorbars=True)

# Viz slices from all 3 axes. 
# Input: 3-elem list, where list[i] = slices to display from axis i
def viz_multi_axes(vol, axes_idxs=[None, None, None], do_plot = False, **kwargs):
  slices, titles = [], []
  for ax in (0, 1, 2):
    idxs = axes_idxs[ax]
    if idxs is None:          idxs = get_mid_idx(vol, ax)
    if isinstance(idxs, int): idxs = [idxs]

    titles += [f"Ax {ax}, slice {i}"    for i in idxs]
    slices += [np.take(vol, i, axis=ax) for i in idxs]

  # plot the slices
  if do_plot: ne.plot.slices(slices, titles=titles, **kwargs)

  return slices, titles

def viz_objs(*objs, do_plot = True, **kwargs):
  # return flattened slices, titles
  slices, titles = zip(*[viz_multi_axes(sitk.GetArrayViewFromImage(o)) for o in objs])
  flat_slices = [s for ax in slices for s in ax]
  flat_titles = [t for ax in titles for t in ax]

  # plot the slices
  if do_plot: ne.plot.slices(flat_slices, titles=flat_titles, **kwargs)

  return flat_slices, flat_titles

## Viz axis (w/ overlay)
# viz segm

# identity "do nothing" function
def id(x): return x

# viz slices from a single axis, w/ optional mask overlay
def viz_axis(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, **kwargs):
  n_slices = len(slices)
  
  # set default options if not given
  options = {
    "grid": (1, n_slices),
    "wspace": 0.0,
    "hspace": 0.0,
    "fig_mult": 1,
    "cmap0": "rainbow",
    "cmap1": mcolors.LinearSegmentedColormap.from_list("", ["white", "yellow"]),
    "cmap2": mcolors.LinearSegmentedColormap.from_list("", ["white", "blue"]),
    "alpha1": 0.7,
    "alpha2": 0.7,
    "axis_fn": id,
  }

  options.update(kwargs)

  axis_fn      = options["axis_fn"]
  nrows, ncols = options["grid"]
  fig_mult     = options["fig_mult"]

  # from SO: https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
  
  fig = plt.figure(figsize=(fig_mult*(ncols+1), fig_mult*(nrows+1))) 
  gs  = gridspec.GridSpec(nrows, ncols,
    wspace=options["wspace"], hspace=options["hspace"], 
    top=1.-0.5/(nrows+1), bottom=0.5/(nrows+1), 
    left=0.5/(ncols+1), right=1-0.5/(ncols+1)) 

  # plot each slice idx
  index = 0
  for row in range(nrows):
    for col in range(ncols):
      ax = plt.subplot(gs[row,col])
      ax.set_title(f"Slice {slices[index]}")
      
      # show ticks only on 1st im
      if index != 0:
        ax.set_xticks([])
        ax.set_yticks([])

      # in case slices in grid > n_slices
      if index < n_slices: 
        ax.imshow(axis_fn(np.take(np_arr, slices[index], fixed_axis)), cmap=options["cmap0"])
        
        # overlay binary mask if provided
        if bin_mask_arr is not None:
          ax.imshow(axis_fn(np.take(bin_mask_arr, slices[index], fixed_axis)), cmap=options["cmap1"], alpha=options["alpha1"])

        # overlay binary mask if provided
        if bin_mask_arr2 is not None:
          ax.imshow(axis_fn(np.take(bin_mask_arr2, slices[index], fixed_axis)), cmap=options["cmap2"], alpha=options["alpha2"])

      else: 
        ax.imshow(np.full((1,1,3), 255)) # show default white image of size 1x1
      
      index += 1
  
  plt.show()
  # return plt

## Misc
# round all floats in a tuple to 3 decimal places
def round_tuple(t, d=3): return tuple(round(x,d) for x in t)

# returns range obj as list
def lrange(a,b): return list(range(a,b))

## Np/SITK conv

# see which slices contain ROI
def get_roi_range(bin_mask_arr, axis):
  slices = np.unique(np.nonzero(bin_mask_arr)[axis])
  return min(slices), max(slices)

# sitk obj and np array have different index conventions
def sitk2np(obj): return np.swapaxes(sitk.GetArrayViewFromImage(obj), 0, 2)

# numpy mask arr into sitk obj
def np2sitk(mask_arr, sitk_image):
  # convert bool mask to int mask
  # swap axes for sitk
  obj = sitk.GetImageFromArray(np.swapaxes(mask_arr.astype(int), 0, 2))
  obj.SetOrigin(sitk_image.GetOrigin())
  obj.SetSpacing(sitk_image.GetSpacing())   
  obj.SetDirection(sitk_image.GetDirection())
  return obj

# In[3]:


# auto-reload when local helper fns change
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data

# In[4]:


PROJ_PATH = "."

# Load train data
train_path = f"{PROJ_PATH}/train_data/train_data"

# Folders containing MR train data
train_folders = os.listdir(train_path)
print(f"Train data folders: {train_folders}")

# make a dictionary of key = train folder, value = (segm obj, nii file)
train_data_dict = {}
for folder in train_folders:
  segm_obj_path = os.path.join(train_path, folder, "Segmentation.obj")

  mp_path      = os.path.join(train_path, folder, "MP-RAGE")
  folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
  folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])
  nii_path     = glob.glob(f"{folder2_path}/*.nii")[0] #os.path.join(folder2_path, os.listdir(folder2_path)[0])
  train_data_dict[folder] = (segm_obj_path, nii_path)
    
# print train data dict
print(f"Training data: key = train folder, value = full path to (segm obj, nii file)\n")
print(*list(train_data_dict.items()), sep="\n")


# ## Moving Image

# In[5]:


# Get path to MR file

moving_folder = "50455"
moving_segm_path, moving_file = train_data_dict[moving_folder]

print(f"Folder: {moving_folder}, MR nii path: {os.path.basename(moving_file)}.")

# compile MR obj from nii file using Simple ITK reader
moving_obj        = sitk.ReadImage(moving_file)
moving_segm       = meshio.read(moving_segm_path)

print("#"*10, f"Moving", "#"*10)
print(f"mr vol shape: {moving_obj.GetSize()}")
print(f"mr vol spacing: {round_tuple(moving_obj.GetSpacing())}")
print(f"mr orientation: {round_tuple(moving_obj.GetDirection())}")


# ### Moving Image Binary Mask

# In[6]:


moving_mask_arr = seg2mask(moving_obj, moving_segm)

# see which slices contain ROI
print(f"ROI contains {np.count_nonzero(moving_mask_arr)} elements")
print(f"ROI non-zero range: ", get_roi_range(moving_mask_arr, axis=0))


# ### Viz Moving Image + Mask

# In[7]:


# see which slices contain ROI
print(f"ROI contains {np.count_nonzero(moving_mask_arr)} elements")
print(f"ROI non-zero range: ", get_roi_range(moving_mask_arr, axis=0))


# In[9]:


viz_axis(sitk2np(moving_obj), bin_mask_arr=moving_mask_arr, 
        slices=lrange(63, 68) + lrange(90,95), fixed_axis=0, \
        axis_fn = np.rot90, \
        grid = [2, 5], hspace=0.3, fig_mult=2)


# In[ ]:





# ## Fixed Image

# In[10]:


# Get path to MR file

fixed_folder = "50456"
fixed_segm_path, fixed_file = train_data_dict[fixed_folder]

print(f"Folder: {fixed_folder}, MR nii path: {os.path.basename(fixed_file)}.")

# compile MR obj from nii file using Simple ITK reader
fixed_obj        = sitk.ReadImage(fixed_file)
fixed_segm       = meshio.read(fixed_segm_path)

print("#"*10, f"Fixed", "#"*10)
print(f"mr vol shape: {fixed_obj.GetSize()}")
print(f"mr vol spacing: {round_tuple(fixed_obj.GetSpacing())}")
print(f"mr orientation: {round_tuple(fixed_obj.GetDirection())}")


# ### Fixed Image Binary Mask

# In[11]:


# get segm as np binary mask arr
fixed_mask_arr = seg2mask(fixed_obj, fixed_segm)

# see which slices contain ROI
print(f"ROI contains {np.count_nonzero(fixed_mask_arr)} elements")
print(f"ROI non-zero range: ", get_roi_range(fixed_mask_arr, axis=0))


# In[12]:


print(f"Fixed ROI non-zero range: ", get_roi_range(fixed_mask_arr, axis=0))
print(f"Moving ROI non-zero range: ", get_roi_range(moving_mask_arr, axis=0))


# # Overlay Moving and Fixed Image

# In[20]:


# Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html
def np_alpha_blend(image1, image2, alpha = 0.5, mask1=None,  mask2=None):
    '''
    Alaph blend two np arr (images), pixels are scalars.
    The region that is alpha blended is controled by the given masks.
    '''
    
    if not mask1: mask1 = np.ones_like(image1)
    if not mask2: mask2 = np.ones_like(image2)
     
    intersection_mask = mask1*mask2
    
    intersection_image = alpha    *intersection_mask * image1 +                          (1-alpha)*intersection_mask * image2
    
    return intersection_image +            mask2-intersection_mask * image2 +            mask1-intersection_mask * image1


# In[21]:


fixed_arr  = sitk2np(fixed_obj)
moving_arr = sitk2np(moving_obj)

full_blended_arr = np_alpha_blend(fixed_arr, moving_arr)


# In[23]:


viz_axis(full_blended_arr,        slices=lrange(63, 68) + lrange(90,95),         fixed_axis=0,         axis_fn = np.rot90,         grid = [2, 5], hspace=0.3, fig_mult=2)


# In[ ]:


# Obtain foreground masks for the two images using Otsu thresholding, we use these later on.
msk1 = sitk.OtsuThreshold(img1,0,1)
msk2 = sitk.OtsuThreshold(img2,0,1)


# # Elastix Registration

# In[ ]:


parameterMap = sitk.GetDefaultParameterMap('translation')

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixed_obj)
elastixImageFilter.SetMovingImage(moving_obj)
elastixImageFilter.SetParameterMap(parameterMap)
elastixImageFilter.Execute()

result_obj = elastixImageFilter.GetResultImage()
transformParameterMap = elastixImageFilter.GetTransformParameterMap()


# ## Tfm Map
# 
# In Elastix, the convention is that resampling from moving=>fixed image domain requires a transformation $T$ from fixed => moving image domain. $T$ maps coordinates in the fixed image domain to the corresponding coordinates in the moving image. Resampling a moving image onto the fixed image coordinate system involves:
# 1. Apply T to voxel coordinates in the **fixed** image to get corresponding coordinates in the **moving** domain: $y = T(x) \in I_M$.
# 2. Get the voxel intensities corresponding to points $y \in I_M$ via interpolation (linear).
# 3. Set the voxel intensities at the fixed image coordinates $x \in I_F$ to the above moving image voxel intensities.

# In[ ]:


transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetTransformParameterMap(transformParameterMap)

population = ['image1.hdr', 'image2.hdr', ... , 'imageN.hdr']

for filename in population:
    transformixImageFilter.SetMovingImage(sitk.ReadImage(filename))
    transformixImageFilter.Execute()
    sitk.WriteImage(transformixImageFilter.GetResultImage(), "result_"+filename)

resultImage


# In[ ]:


result_arr = sitk.GetArrayFromImage(resultImage)


# In[ ]:


np_result_arr = sitk2np(resultImage)


# In[ ]:


viz_axis(np_result_arr, bin_mask_arr=moving_mask_arr, 
        slices=lrange(63, 68) + lrange(90,95), fixed_axis=0, \
        axis_fn = np.rot90, \
        grid = [2, 5], hspace=0.3, fig_mult=2)


# In[ ]:


import numpy as np


# In[ ]:


print(np.array([1,2,3]))


# In[ ]:




