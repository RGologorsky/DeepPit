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