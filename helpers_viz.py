# numpy and SITK
import numpy as np
import SimpleITK as sitk

# viz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec

# identity "do nothing" function
def id(x): return x

# rescale np arr values to be 0-255 (RGB range)
def rgb_scale(np_arr): 
    return ((np_arr - np_arr.min())/(np_arr.max() - np_arr.min()) * 255).astype('uint8')

# convert color string to RGB tuple (0-255)
def color2rgb(color): return (np.array(mcolors.to_rgba(color)[:3]) * 255).astype('uint8')

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
  np_arr = rgb_scale(np_arr)
    
  # apply color binary masks
  if bin_mask_arr is not None:
    # filter size
    bin_mask_arr = np.take(bin_mask_arr, slices, fixed_axis)
    
    # add color-coded ROIs
    color1_rgb = options["alpha1"] * color2rgb(options["color1"])
    
    # stack three channels, grayscale => RGB
    np_arr = np.stack((np_arr + bin_mask_arr * color1_rgb[0], \
                       np_arr + bin_mask_arr * color1_rgb[1], \
                       np_arr + bin_mask_arr * color1_rgb[2]),
                       axis = -1)
    
  if bin_mask_arr2 is not None:
    # filter size
    bin_mask_arr2 = np.take(bin_mask_arr2, slices, fixed_axis)
    
    # add RGB components to each channel of np arr
    color2_rgb = options["alpha2"] * color2rgb(options["color2"])
    np_arr += np.stack((bin_mask_arr2 * color2_rgb[0], \
                        bin_mask_arr2 * color2_rgb[1], \
                        bin_mask_arr2 * color2_rgb[2]), \
                       axis=-1) 
    
    
  # rescale RGB; 0-255, uint8 to save memory
  np_arr = rgb_scale(np_arr)
        
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