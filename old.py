# old
def seg2mask(image_path, segmentation_path):

  # import the .nii object
  dicom_img = nib.load(image_path)

  # import the segmentation mesh
  segmentation_mesh = meshio.read(segmentation_path)

  # Compute Delaunay triangulation of points.
  tri = Delaunay(segmentation_mesh.points)

  # define the voxel - realworld mappings 
  voxel_to_realworld_transform = dicom_img.affine
  realworld_to_voxel_transform = np.linalg.inv(voxel_to_realworld_transform)

  # initialize numpy arrays
  dicom_img_numpy_array = np.array(dicom_img.get_fdata())
  binary_segmentation_mask = np.zeros_like(dicom_img_numpy_array, dtype=np.bool_)

  # if you want to spot test a single slice, set the range to "range(80, dicom.shape[0])" this is a slice in the middle of the
  # MRI image which should be segmented. Then, uncomment the show_slices line to see the MRI and the segmentation

  # for readability
  shape0, shape1, shape2 = dicom_img_numpy_array.shape

  # from SO: https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points
  # equiv: np.array([(i,j,k) for i in range(shape0) for j in range(shape1) for k in range(shape2)])
  #voxel_location_array = np.array(np.meshgrid(range(shape0), range(shape1), range(shape2), indexing='ij')).T.reshape(-1, 3)[:,[2,1,0]]
  voxel_location_array = np.indices((shape2, shape1, shape0)).T.reshape(-1,3)[:,[2,1,0]]
  realworld_location_array = apply_affine(voxel_to_realworld_transform, voxel_location_array)
  binary_segmentation_mask = (tri.find_simplex(realworld_location_array) >= 0).reshape(shape0, shape1, shape2)

  return dicom_img_numpy_array, binary_segmentation_mask

def viz_axis(np_arr, slices, fixed_axis, bin_mask_arr=None, bin_mask_arr2 = None, **kwargs):
  n_slices = len(slices)
  
  # set default options if not given
  options = {
    "grid": (1, n_slices),
    "wspace": 0.0,
    "hspace": 0.0,
    "fig_mult": 1,
    "cmap0": plt.cm.gray, #"rainbow",
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
            ax.imshow(axis_fn(np.take(bin_mask_arr, slices[index], fixed_axis)), alpha=options["alpha1"])
            #ax.imshow(axis_fn(np.take(bin_mask_arr, slices[index], fixed_axis)), cmap=options["cmap1"], alpha=options["alpha1"])

        # overlay binary mask if provided
        if bin_mask_arr2 is not None:
          ax.imshow(axis_fn(np.take(bin_mask_arr2, slices[index], fixed_axis)), cmap=options["cmap2"], alpha=options["alpha2"])

      else: 
        ax.imshow(np.full((1,1,3), 255)) # show default white image of size 1x1
      
      index += 1
  
  plt.show()
  # return plt

# to save memory
def scale_to_255(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return ((arr - arr_min)/(arr_max - arr_min) * 255).astype('uint8')
        
def gray_to_255rgb(arr): return np.stack((scale_to_255(arr),)*3, axis = -1)
def color_to_255rgb(color): return (np.array(mcolors.to_rgba(color)[:3]) * 255).astype('uint8')

# scale between [0,1]                                                                                   
def unit_scale(arr): 
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min)/(arr_max - arr_min)

def gray2unit_rgb(arr): return np.stack((unit_scale(arr),)*3, axis = -1)
def color2rgb(color): return np.array(mcolors.to_rgba(color)[:3])
