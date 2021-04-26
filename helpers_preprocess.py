# Preprocess Helper fns
#
#  1. make_affine/segm2mask
#  2. get_isotropic
#  3. get_data_dict
#  4. folder2objs
#  5. mask2bbox / print bbox
#

# load data
import os

# filter filenames for .nii
import glob

# meshio for 3DSlicer segm obj
import meshio

# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk
from helpers_general import np2sitk, sitk2np

# segmentation
from scipy.spatial   import Delaunay

# given folder name, return isotropic SITK obj of nii and segm obj
def folder2objs(folder_name, train_data_dict, ras_adj = False):
    segm_path, file = train_data_dict[folder_name]

    # compile MR obj from nii file using Simple ITK reader
    obj        = sitk.ReadImage(file)
    segm       = meshio.read(segm_path)
    mask_arr   = seg2mask(obj, segm, ras_adj)
    
    return obj, np2sitk(mask_arr, obj)

# Convert segmentation object to numpy binary mask
# 1. Get affine matrix in SITK (aff tfm: idx coord => physical space coord)
# 2. Convert image idxs to physical coords
# 3. Check whether physical coords are in the Delauney triangulation of segmented mesh points

# 1. Get affine matrix in SITK
# https://niftynet.readthedocs.io/en/v0.2.2/_modules/niftynet/io/simple_itk_as_nibabel.html
def make_affine(simpleITKImage, ras_adj):
    # get affine transform in LPS
    c = [simpleITKImage.TransformIndexToPhysicalPoint(p)
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
    if ras_adj:
        affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine

# Seg2mask
def seg2mask(image_obj, segm_obj, ras_adj):
    dims = image_obj.GetSize()
    aff     = make_affine(image_obj, ras_adj)
    idx_pts = np.indices(dims[::-1], dtype=np.uint16).T.reshape(-1,3)[:,[2,1,0]]
    physical_pts = (np.dot(aff[:3,:3], idx_pts.T) + aff[:3,3:4]).T 
    return (Delaunay(segm_obj.points).find_simplex(physical_pts) >= 0).reshape(dims)


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


# make a dictionary of key = train folder, value = (segm obj, nii file)
def get_data_dict(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
      segm_obj_path = os.path.join(train_path, folder, "Segmentation.obj")

      mp_path      = os.path.join(train_path, folder, "MP-RAGE")
      folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
      folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])
      nii_path     = glob.glob(f"{folder2_path}/*.nii")[0] #os.path.join(folder2_path, os.listdir(folder2_path)[0])
      train_data_dict[folder] = (segm_obj_path, nii_path)
    return train_data_dict

# # given folder name, return isotropic SITK obj of nii and segm obj
# def folder2objs(folder_name, train_data_dict, ras_adj = False):
#     segm_path, file = train_data_dict[folder_name]

#     # compile MR obj from nii file using Simple ITK reader
#     obj        = sitk.ReadImage(file)
#     segm       = meshio.read(segm_path)
#     mask_arr   = seg2mask(obj, segm, ras_adj)
    
#     return obj, np2sitk(mask_arr, obj)

# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
# https://stackoverflow.com/questions/39206986/numpy-get-rectangle-area-just-the-size-of-mask/48346079
def mask2bbox(mask):

    i = np.any(mask, axis=(1, 2))
    j = np.any(mask, axis=(0, 2))
    k = np.any(mask, axis=(0, 1))

    imin, imax = np.where(i)[0][[0, -1]]
    jmin, jmax = np.where(j)[0][[0, -1]]
    kmin, kmax = np.where(k)[0][[0, -1]]

    return imin, imax, jmin, jmax, kmin, kmax

def get_bbox_size(imin, imax, jmin, jmax, kmin, kmax):
    return {imax - imin}, {jmax-jmin}, {kmax-kmin}

def print_bbox_size(imin, imax, jmin, jmax, kmin, kmax):
    print(f"{imax - imin}, {jmax-jmin}, {kmax-kmin}")
    
def print_bbox(imin, imax, jmin, jmax, kmin, kmax):
    print(f"Bbox coords: ({imin}, {jmin}, {kmin}) to ({imax}, {jmax}, {kmax}). Size: {imax - imin}, {jmax-jmin}, {kmax-kmin}.")
    print(f"Bounding box coord: from location ({jmin}, {kmin}) of slice {imin} to location ({jmax}, {kmax}) of slice {imax}.")
    #print(f"Slices: {imin}, {imax} ({imax-imin}), Rows: {jmin}, {jmax} ({jmax-jmin}), Cols: {kmin}, {kmax} ({kmax-kmin}).")
    
    
    
# given folder name, return isotropic SITK obj of nii and segm obj
def folder2objs_old(folder_name, train_data_dict, iso_spacing = (1, 1, 1), iso_interpolator = sitk.sitkLinear, ras_adj = False):
    segm_path, file = train_data_dict[folder_name]

    # compile MR obj from nii file using Simple ITK reader
    obj        = sitk.ReadImage(file)
    segm       = meshio.read(segm_path)
    mask_arr   = seg2mask(obj, segm, ras_adj)
    
    # preprocess
    
    # 1. isotropic
    iso_obj       = get_isotropic(obj, iso_spacing, iso_interpolator)
    iso_mask_obj  = get_isotropic(np2sitk(mask_arr, obj), iso_spacing, iso_interpolator)
    
    return iso_obj, iso_mask_obj