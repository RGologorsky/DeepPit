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
    
# return np arr representing viz slices from a single axis, w/ optional mask overlay
def get_viz_arr(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, \
                color1 = None, color2 = None, alpha1 = None, alpha2 = None, \
                crop_coords = None, crop_extra = None):
    
  n_slices = len(slices)

  # 1. filter size down to slices of interest
  # 2. within slices, crop inputs to bbox (+- extra area around bbox for context)
    
  #  crop to fixed axis' slices of interest to be viz
  np_arr = np.take(np_arr, slices, fixed_axis)
  if bin_mask_arr is not None:   bin_mask_arr  = np.take(bin_mask_arr, slices, fixed_axis)
  if bin_mask_arr2 is not None:  bin_mask_arr2 = np.take(bin_mask_arr2, slices, fixed_axis)
        
  # if cropping, filter down to crop area (+ extra)
  if crop_coords is not None:
    pad = crop_extra
    imin, imax, jmin, jmax, kmin, kmax = crop_coords
    if fixed_axis == 0:   
        jmin -= pad; jmax += pad; kmin -= pad; kmax += pad;
        np_arr = np_arr[:, jmin:jmax, kmin:kmax]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[:, jmin:jmax, kmin:kmax]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[:, jmin:jmax, kmin:kmax]
        
    elif fixed_axis == 1: 
        imin -= pad; imax += pad; kmin -= pad; kmax += pad;
        np_arr = np_arr[imin:imax, :, kmin:kmax]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[imin:imax, :, kmin:kmax]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[imin:imax, :, kmin:kmax]
        
    else:
        imin -= pad; imax += pad; jmin -= pad; jmax += pad;
        np_arr = np_arr[imin:imax, jmin:jmax, :]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[imin:imax, jmin:jmax, :]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[imin:imax, jmin:jmax, :]
    
  # rescale RGB; 0-255, uint8 to save memory
  np_arr = rgb_scale(np_arr)
    
  # apply color binary masks
  if bin_mask_arr is not None:              
    # add color-coded ROIs
    color1_rgb = alpha1 * color2rgb(color1)
    
    # stack three channels, grayscale => RGB
    np_arr = np.stack((np_arr + bin_mask_arr * color1_rgb[0], \
                       np_arr + bin_mask_arr * color1_rgb[1], \
                       np_arr + bin_mask_arr * color1_rgb[2]),
                       axis = -1)
    
  if bin_mask_arr2 is not None:
    # add RGB components to each channel of np arr
    color2_rgb = alpha2 * color2rgb(color2)
    np_arr += np.stack((bin_mask_arr2 * color2_rgb[0], \
                        bin_mask_arr2 * color2_rgb[1], \
                        bin_mask_arr2 * color2_rgb[2]), \
                       axis=-1) 
    
    
  # rescale RGB; 0-255, uint8 to save memory
  return rgb_scale(np_arr)
    
# viz slices from a single axis, w/ optional mask overlay
def viz_axis(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, **kwargs):
    n_slices = len(slices)
    
    # set default options if not given
    fig_options = {
        "grid": (1, n_slices),
        "wspace": 0.0,
        "hspace": 0.0,
        "fig_mult": 1,
        "cmap0": None, #plt.cm.gray, #"rainbow", 
    }

    img_options = {
        "color1": "yellow",
        "color2": "blue",
        "alpha1": 0.3,
        "alpha2": 0.3,
        "crop_coords": None,
        "crop_extra": 0,
    }


    img_options2 = {
        "color1_b": "yellow",
        "color2_b": "blue",
        "alpha1_b": 0.3,
        "alpha2_b": 0.3,
        "crop_coords_b": None,
        "crop_extra_b": 0,
    }
    
    # update kwargs last
    options = {"axis_fn": id, "axis_fn_b": id, **fig_options, **img_options, **img_options2}
    options.update(kwargs)

    # 1. filter size down to slices of interest
    # 2. within slices, crop inputs to bbox (+- extra area around bbox for context)
    keys = ["np_arr", "slices", "fixed_axis", "bin_mask_arr", "bin_mask_arr2", \
          "color1", "color2", "alpha1", "alpha2", "crop_coords", "crop_extra"]
             
    np_arr = get_viz_arr(np_arr, slices, fixed_axis, bin_mask_arr, bin_mask_arr2, \
                         *[options.get(k, None) for k in keys[-6:]])

    if "np_arr_b" in options:
        np_arr2 = get_viz_arr(*[options.get(k+"_b", None) for k in keys])

    # from SO: https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    axis_fn      = options["axis_fn"]
    fig_mult     = options["fig_mult"]

    nrows, ncols = options["grid"]
    
    nrows1 = n_slices//ncols
    nrows2 = nrows-nrows1

    fig = plt.figure(figsize=(fig_mult*(ncols+1), fig_mult*(nrows+1))) 
    gs  = gridspec.GridSpec(nrows, ncols,
    wspace=options["wspace"], hspace=options["hspace"], 
    top=1.-0.5/(nrows+1), bottom=0.5/(nrows+1), 
    left=0.5/(ncols+1), right=1-0.5/(ncols+1)) 

    # plot each slice idx
    index = 0                           
    for row in range(nrows1):
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
    # 2nd set
    axis_fn2 = options.get("axis_fn_b", None)
    slices2  = options.get("slices_b", None)
    fixed_axis2 = options.get("fixed_axis_b", None)
    
    index = 0                           
    for row in range(nrows2):
        for col in range(ncols):
            ax = plt.subplot(gs[nrows1 + row,col])
            ax.set_title(f"Slice {slices2[index]} (#2)")

            # show ticks only on 1st im
            if index != 0:
                ax.set_xticks([])
                ax.set_yticks([])

            # in case slices in grid > n_slices
            if index < n_slices: 
                ax.imshow(axis_fn2(np.take(np_arr2, index, fixed_axis2)), cmap=options["cmap0"])
            else: 
                ax.imshow(np.full((1,1,3), 255)) # show default white image of size 1x1

            index += 1
    plt.show()
    # return plt

# viz slices from a single axis, w/ optional mask overlay
def viz_get_np_arr(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, crop_coords = None, crop_extra = 0):
  n_slices = len(slices)

  # 1. filter size down to slices of interest
  # 2. within slices, crop inputs to bbox (+- extra area around bbox for context)
    
  #  crop to fixed axis' slices of interest to be viz
  np_arr = np.take(np_arr, slices, fixed_axis)
  if bin_mask_arr is not None:   bin_mask_arr  = np.take(bin_mask_arr, slices, fixed_axis)
  if bin_mask_arr2 is not None:  bin_mask_arr2 = np.take(bin_mask_arr2, slices, fixed_axis)
        
  # if cropping, filter down to crop area (+ extra)
  if options["crop_coords"] is not None:
    pad = options["crop_extra"]
    imin, imax, jmin, jmax, kmin, kmax = options["crop_coords"]
    if fixed_axis == 0:   
        jmin -= pad; jmax += pad; kmin -= pad; kmax += pad;
        np_arr = np_arr[:, jmin:jmax, kmin:kmax]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[:, jmin:jmax, kmin:kmax]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[:, jmin:jmax, kmin:kmax]
        
    elif fixed_axis == 1: 
        imin -= pad; imax += pad; kmin -= pad; kmax += pad;
        np_arr = np_arr[imin:imax, :, kmin:kmax]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[imin:imax, :, kmin:kmax]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[imin:imax, :, kmin:kmax]
        
    else:
        imin -= pad; imax += pad; jmin -= pad; jmax += pad;
        np_arr = np_arr[imin:imax, jmin:jmax, :]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[imin:imax, jmin:jmax, :]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[imin:imax, jmin:jmax, :]
    
  # rescale RGB; 0-255, uint8 to save memory
  np_arr = rgb_scale(np_arr)
    
  # apply color binary masks
  if bin_mask_arr is not None:              
    # add color-coded ROIs
    color1_rgb = options["alpha1"] * color2rgb(options["color1"])
    
    # stack three channels, grayscale => RGB
    np_arr = np.stack((np_arr + bin_mask_arr * color1_rgb[0], \
                       np_arr + bin_mask_arr * color1_rgb[1], \
                       np_arr + bin_mask_arr * color1_rgb[2]),
                       axis = -1)
    
  if bin_mask_arr2 is not None:
    # add RGB components to each channel of np arr
    color2_rgb = options["alpha2"] * color2rgb(options["color2"])
    np_arr += np.stack((bin_mask_arr2 * color2_rgb[0], \
                        bin_mask_arr2 * color2_rgb[1], \
                        bin_mask_arr2 * color2_rgb[2]), \
                       axis=-1) 
    
    
  # rescale RGB; 0-255, uint8 to save memory
  return rgb_scale(np_arr)