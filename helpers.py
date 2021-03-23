# Helper fns

# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk

# segmentation
from scipy.spatial   import Delaunay

# Convert segmentation object to numpy binary mask
# 1. Get affine matrix in SITK (aff tfm: idx coord => physical space coord)
# 2. Convert image idxs to physical coords
# 3. Check whether physical coords are in the Delauney triangulation of segmented mesh points

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
    idx_pts = np.indices(dims[::-1], dtype=np.uint16).T.reshape(-1,3)[:,[2,1,0]]
    physical_pts = (np.dot(aff[:3,:3], idx_pts.T) + aff[:3,3:4]).T 
    return (Delaunay(segm_obj.points).find_simplex(physical_pts) >= 0).reshape(dims)


## Misc

# round all floats in a tuple to 3 decimal places
def round_tuple(t, d=3): return tuple(round(x,d) for x in t)

# returns range obj as list
def lrange(a,b): return list(range(a,b))

# SITK

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
  obj = sitk.GetImageFromArray(np.swapaxes(mask_arr.astype(np.uint8), 0, 2))
  obj.SetOrigin(sitk_image.GetOrigin())
  obj.SetSpacing(sitk_image.GetSpacing())   
  obj.SetDirection(sitk_image.GetDirection())
  return obj

# isotropic
def get_isotropic(obj, new_spacing = (1,1,1), interpolator=sitk.sitkLinear):
  """ returns obj w/ 1mm isotropic voxels """

  original_size    = obj.GetSize()
  original_spacing = obj.GetSpacing()

  min_spacing = min(new_spacing)

  new_size = [int(round(osz*ospc/min_spacing)) for osz,ospc in zip(original_size, original_spacing)]

  return sitk.Resample(obj, new_size, sitk.Transform(), interpolator,
                         obj.GetOrigin(), new_spacing, obj.GetDirection(), 0,
                         obj.GetPixelID())