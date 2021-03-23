# 1. Convert image indices to physical location
# 2. Check if physical locations are within Delaney triangulated mesh points represeting the segmentation
def new_seg2mask(image_obj, segmentation_path):
  ''' input: image obj and segm obj
      output: segm as numpy binary arr
  '''
  voxel_location_array = np.indices((shape2, shape1, shape0)).T.reshape(-1,3)[:,[2,1,0]]
  realworld_location_array = image_obj.GetOrigin() + \
                             image_obj.GetDirection() * image_obj.GetSpacing() * voxel_location_array

  return Delaunay(meshio.read(segmentation_path).points).find_simplex(realworld_location_array) >= 0