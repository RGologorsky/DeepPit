# numpy and SITK
import numpy as np
import SimpleITK as sitk

# viz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec

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

# Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/05_Results_Visualization.html
def np_alpha_blend(image1, image2, alpha = 0.5, mask1=None,  mask2=None):
    '''
    Alaph blend two np arr (images), pixels are scalars.
    The region that is alpha blended is controled by the given masks.
    '''
    
    if mask1 is not None: mask1 = np.ones_like(image1)
    if mask2 is not None: mask2 = np.ones_like(image2)
     
    intersection_mask = mask1*mask2
    
    intersection_image = alpha    *intersection_mask * image1 + \
                         (1-alpha)*intersection_mask * image2
    
    return intersection_image + \
           mask2-intersection_mask * image2 + \
           mask1-intersection_mask * image1

# viz slices from a single axis, w/ optional mask overlay
def viz_axis(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, **kwargs):
  n_slices = len(slices)
  
  # set default options if not given
  options = {
    "grid": (1, n_slices),
    "wspace": 0.0,
    "hspace": 0.0,
    "fig_mult": 1,
    "cmap0": None, #plt.cm.gray, #"rainbow",
    "color1": "yellow",
    "color2": "blue",
    "alpha1": 0.3,
    "alpha2": 0.3,
    "axis_fn": id,
  }

  options.update(kwargs)

  axis_fn      = options["axis_fn"]
  nrows, ncols = options["grid"]
  fig_mult     = options["fig_mult"]
  
  # filter size down to slices of interest
  np_arr = np.take(np_arr, slices, fixed_axis)
  
  # rescale RGB; 0-255, uint8 to save memory
  np_arr = ((np_arr - np_arr.min())/(np_arr.max() - np_arr.min()) * 255).astype('uint8')
    
  # apply color binary masks
  if bin_mask_arr is not None:
    # filter size
    bin_mask_arr = np.take(bin_mask_arr, slices, fixed_axis)
    
    # add color-coded ROIs
    color1_rgb = options["alpha1"] * color_to_255rgb(options["color1"])
    
    np_arr = np.stack((np_arr + bin_mask_arr * color1_rgb[0], \
                       np_arr + bin_mask_arr * color1_rgb[1], \
                       np_arr + bin_mask_arr * color1_rgb[2]),
                       axis = -1)
    
  if bin_mask_arr2 is not None:
    # filter size
    bin_mask_arr2 = np.take(bin_mask_arr2, slices, fixed_axis)
    
    color2_rgb = options["alpha2"] * color_to_255rgb(options["color2"])
    np_arr += np.stack((bin_mask_arr2 * color2_rgb[0], \
                        bin_mask_arr2 * color2_rgb[1], \
                        bin_mask_arr2 * color2_rgb[2]), \
                       axis=-1) 
    
    
  # rescale RGB; 0-255, uint8 to save memory
  np_arr = ((np_arr - np_arr.min())/(np_arr.max() - np_arr.min()) * 255).astype('uint8')
        
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
        ax.imshow(axis_fn(np.take(np_arr, index, fixed_axis)), cmap=options["cmap0"])
      else: 
        ax.imshow(np.full((1,1,3), 255)) # show default white image of size 1x1
      
      index += 1
  
  plt.show()
  # return plt