# vix_axis, viz_bbox(mr, seg, pred)
from helpers.preprocess import mask2bbox, print_bbox

# numpy and SITK
import torch
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

# get figure
def get_figure_gridspec(nrows, ncols, wspace=0.0, hspace=0.0, fig_mult=1):
    fig = plt.figure(figsize=(fig_mult*(ncols+1), fig_mult*(nrows+1))) 
    gs  = gridspec.GridSpec(nrows, ncols,
    wspace=options["wspace"], hspace=options["hspace"], 
    top=1.-0.5/(nrows+1), bottom=0.5/(nrows+1), 
    left=0.5/(ncols+1), right=1-0.5/(ncols+1))
    
    return fig, gs
    
# return np arr representing viz slices from a single axis, w/ optional mask overlay
def get_viz_arr(np_arr, slices, fixed_axis, bin_mask_arr, bin_mask_arr2, \
                color1, color2, alpha1, alpha2, \
                crop_coords, crop_extra):
    
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
    
    shape0, shape1, shape2 = np_arr.shape
        
    if fixed_axis == 0:   
        jmin -= pad; jmax += pad; kmin -= pad; kmax += pad;

        jmin = max(0, jmin); kmin = max(0, kmin)
        jmax = min(jmax,shape1); kmax = min(kmax, shape2)

        np_arr = np_arr[:, jmin:jmax, kmin:kmax]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[:, jmin:jmax, kmin:kmax]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[:, jmin:jmax, kmin:kmax]

    elif fixed_axis == 1: 
        imin -= pad; imax += pad; kmin -= pad; kmax += pad;

        imin = max(0, imin); kmin = max(0, kmin)
        imax = min(imax, shape0); kmax = min(kmax, shape2)

        np_arr = np_arr[imin:imax, :, kmin:kmax]
        if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[imin:imax, :, kmin:kmax]
        if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[imin:imax, :, kmin:kmax]

    else:
        imin -= pad; imax += pad; jmin -= pad; jmax += pad;

        imin = max(0, imin); jmin = max(0, jmin)
        imax = min(imax, shape0); jmax = min(jmax, shape1)

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

# https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b): return -(-a // b)

# viz slices from a single axis, w/ optional mask overlay
# np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, **kwargs
def viz_axis(**kwargs):
    
    # set default nrows, ncols = grid
    if "grid" in kwargs:
        grid = kwargs["grid"]
        nrows, ncols = grid    
    else:
        # set default number of cols
        ncols = kwargs.get("ncols", len(kwargs["slices"]))
       
        # set default number of rows
        n_per_row = [len(kwargs[k]) for k in kwargs.keys() if k.startswith("slices")]
        n_per_row = [ceildiv(n,ncols) for n in n_per_row]
        nrows = sum(n_per_row)
    
    # set default options if not given
    fig_options = {
        "wspace": 0.0,
        "hspace": 0.0,
        "fig_mult": 1,
        "cmap0": None, #plt.cm.gray, #"rainbow", 
    }

    img_options = {
        "np_arr": None,
        "slices": None,
        "fixed_axis": None,
        "bin_mask_arr": None,
        "bin_mask_arr2": None,
        "axis_fn": id,
        "color1": "yellow",
        "color2": "blue",
        "alpha1": 0.3,
        "alpha2": 0.3,
        "crop_coords": None,
        "crop_extra": 0,
    }
    
    # update kwargs last
    options = {**fig_options, **img_options}
    options.update(kwargs)
    
    # from SO: https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    fig_mult = options["fig_mult"]    
    fig = plt.figure(figsize=(fig_mult*(ncols+1), fig_mult*(nrows+1))) 
    gs  = gridspec.GridSpec(nrows, ncols,
    wspace=options["wspace"], hspace=options["hspace"], 
    top=1.-0.5/(nrows+1), bottom=0.5/(nrows+1), 
    left=0.5/(ncols+1), right=1-0.5/(ncols+1)) 
    
    np_keys = ["np_arr", "slices", "fixed_axis", "bin_mask_arr", "bin_mask_arr2", \
              "color1", "color2", "alpha1", "alpha2", "crop_coords", "crop_extra"]
    
    #np_keys_no_default = ["np_arr", "bin_mask_arr", "bin_mask_arr2"]
    
    nrows_offset = 0
    np_arr_count = 0
    
    for key in kwargs.keys():
        if key.startswith("np_arr"): 
            np_arr_count += 1
            
            # 1. filter size down to slices of interest
            # 2. within slices, crop inputs to bbox (+- extra area around bbox for context)

            suffix = key[6:]
            #print(key, suffix, options["color1" + suffix], options["color1"]) 
            np_arr = get_viz_arr(*[options.get(k+suffix, None if k == "bin_mask_arr2" else options[k]) for k in np_keys])

            axis_fn    = options.get("axis_fn"    + suffix, options["axis_fn"])
            slices     = options.get("slices"     + suffix, options["slices"])
            fixed_axis = options.get("fixed_axis" + suffix, options["fixed_axis"])
            
            title      = options.get("title"      + suffix, f"{np_arr_count}")

            np_nrows = ceildiv(len(slices), ncols)

            # plot each slice idx
            index = 0                           
            for row in range(np_nrows):
                for col in range(ncols):
                    ax = plt.subplot(gs[nrows_offset + row,col])

                    # show ticks only on 1st im
                    if index != 0:
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # in case slices in grid > n_slices
                    if index < len(slices): 
                        ax.imshow(axis_fn(np.take(np_arr, index, fixed_axis)), cmap=options["cmap0"])
                        ax.set_title(f"Slice {slices[index]} ({title})")
                    else: 
                        ax.imshow(np.full((1,1,3), 255)) # show default white image of size 1x1
                        ax.set_title(f"NA")

                    index += 1
            nrows_offset += np_nrows
    plt.show()
    # return plt

def bbox_union(bbox1, bbox2):
    l1 = [min(bbox1[i], bbox2[i]) for i in (0, 2, 4)]
    l2 = [max(bbox1[i], bbox2[i]) for i in (1, 3, 5)]    
    return [val for pair in zip(l1, l2) for val in pair]

# Viz
def viz_bbox(mr, seg, pred):
#     mr, seg = learn.dls.train_ds[idx] 
#     pred = learn.predict(test_items[idx])[0]
    
    # dice (add B dimension)
#     dice = dice_score(pred.unsqueeze(0), seg.unsqueeze(0).unsqueeze(0))
#     print(f"Dice: {dice:0.4f}")
    
    # convert pred to mask
    pred_mk   = torch.argmax(pred, dim=0)
    pred_bbox = mask2bbox(np.array(pred_mk))

    mr, seg = np.array(mr), np.array(seg)
    gt_bbox = mask2bbox(mr)
    
    # union bbox
    bbox = bbox_union(gt_bbox, pred_bbox)
    
    # print bbox
    print("Pred: "); print_bbox(*pred_bbox)
    print("GT: "); print_bbox(*gt_bbox)
    print("Union: "); print_bbox(*bbox)
          
    # viz
    viz_axis(np_arr = mr, \
            bin_mask_arr   = seg,     color1 = "yellow",  alpha1=0.3, \
            bin_mask_arr2  = pred_mk, color2 = "magenta", alpha2=0.3, \
            slices=lrange(*bbox[0:2]), fixed_axis=0, \
            axis_fn = np.rot90, \
            title   = "Axis 0", \

            np_arr_b = mr, \
            bin_mask_arr_b   = seg,     color1_b = "yellow",  alpha1_b=0.3, \
            bin_mask_arr2_b  = pred_mk, color2_b = "magenta", alpha2_b=0.3, \
            slices_b = lrange(*bbox[2:4]), fixed_axis_b=1, \
            title_b  = "Axis 1", \

            np_arr_c = mr, \
            bin_mask_arr_c   = seg,     color1_c = "yellow",  alpha1_c=0.3, \
            bin_mask_arr2_c  = pred_mk, color2_c = "magenta", alpha2_c=0.3, \
            slices_c = lrange(*bbox[4:6]), fixed_axis_c=2, \
            title_c = "Axis 2", \
  
        ncols = 5, hspace=0.3, fig_mult=2)

# viz slices from a single axis, w/ optional mask overlay
def viz_get_np_arr(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, crop_coords = None, crop_extra = 0):
    print("hello")
    n_slices = len(slices)

    # 1. filter size down to slices of interest
    # 2. within slices, crop inputs to bbox (+- extra area around bbox for context)

    #  crop to fixed axis' slices of interest to be viz
    np_arr = np.take(np_arr, slices, fixed_axis)
    if bin_mask_arr is not None:   bin_mask_arr  = np.take(bin_mask_arr, slices, fixed_axis)
    if bin_mask_arr2 is not None:  bin_mask_arr2 = np.take(bin_mask_arr2, slices, fixed_axis)
        
    # if cropping, filter down to crop area (+ extra)
    if options["crop_coords"] is not None:
        # print("hi")
        pad = options["crop_extra"]
        imin, imax, jmin, jmax, kmin, kmax = options["crop_coords"]
        
        shape0, shape1, shape2 = np_arr.shape
        
        if fixed_axis == 0:   
            jmin -= pad; jmax += pad; kmin -= pad; kmax += pad;
            
            jmin = max(0, jmin); kmin = max(0, kmin)
            jmax = min(0, shape1); kmax = min(0, shape2)
            
            np_arr = np_arr[:, jmin:jmax, kmin:kmax]
            if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[:, jmin:jmax, kmin:kmax]
            if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[:, jmin:jmax, kmin:kmax]
            
            print("hi")
            print(jmin, jmax, kmin, kmax)
            print(np_arr.shape)

        elif fixed_axis == 1: 
            imin -= pad; imax += pad; kmin -= pad; kmax += pad;
            
            imin = max(0, imin); kmin = max(0, kmin)
            imax = min(0, shape0); kmax = min(0, shape2)
            
            np_arr = np_arr[imin:imax, :, kmin:kmax]
            if bin_mask_arr is not None:  bin_mask_arr  = bin_mask_arr[imin:imax, :, kmin:kmax]
            if bin_mask_arr2 is not None: bin_mask_arr2 = bin_mask_arr2[imin:imax, :, kmin:kmax]

        else:
            imin -= pad; imax += pad; jmin -= pad; jmax += pad;
            
            imin = max(0, imin); jmin = max(0, jmin)
            imax = min(0, shape0); jmax = min(0, shape1)
            
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