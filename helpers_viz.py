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