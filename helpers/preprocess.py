# Preprocess Helper fns
#
#  1. make_affine/segm2mask
#  2. get_isotropic
#  3. get_data_dict
#  4. folder2objs
#  5. mask2bbox / print bbox
#  6. crop thershold
#  7. resample2reference
#  8. get largest connected copmonent

# load data
import os

# filter filenames for .nii
import glob

# meshio for 3DSlicer segm obj
import meshio

# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk
import torch # torch mask2bbox
from .general import mask2sitk, sitk2np

# segmentation
from scipy.spatial   import Delaunay
    
# given paths, return isotropic SITK obj of nii and segm obj
def paths2objs(mr_path, segm_path, ras_adj = False):
    mr         = sitk.ReadImage(mr_path, sitk.sitkFloat32)
    segm       = meshio.read(segm_path)
    mask_arr   = seg2mask(mr, segm, ras_adj)
    
    return mr, mask2sitk(mask_arr, mr)

# given folder name, return isotropic SITK obj of nii and segm obj
def folder2objs(folder_name, train_data_dict, ras_adj = False):
    segm_path, file = train_data_dict[folder_name]
    folder2objs(segm_path, file)

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
def get_data_dict_n4(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
        segm_obj_path = os.path.join(train_path, folder, "seg.pt")

        mp_path      = os.path.join(train_path, folder, "MP-RAGE")
        folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
        folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])

        # choose corrected_n4 if available
        nii_paths = glob.glob(f"{folder2_path}/*.nii")
        nii_path = nii_paths[0]
         
        if len(nii_paths) > 1 and not nii_path.endswith("corrected_n4.nii"):
            nii_path = nii_paths[1]
            
        train_data_dict[folder] = (nii_path, segm_obj_path) #(segm_obj_path, nii_path)
    return train_data_dict

# make a dictionary of key = train folder, value = (segm obj, nii file)
def get_data_dict(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
      segm_obj_path = os.path.join(train_path, folder, "seg.pt")

      mp_path      = os.path.join(train_path, folder, "MP-RAGE")
      folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
      folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])
      nii_path     = glob.glob(f"{folder2_path}/*.nii")[0] #os.path.join(folder2_path, os.listdir(folder2_path)[0])
      train_data_dict[folder] = (nii_path, segm_obj_path) #(segm_obj_path, nii_path)
    return train_data_dict

# make a dictionary of key = train folder, value = (segm obj, nii file)
def get_data_dict_old(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
      segm_obj_path = os.path.join(train_path, folder, "Segmentation.obj")

      mp_path      = os.path.join(train_path, folder, "MP-RAGE")
      folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
      folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])
      nii_path     = glob.glob(f"{folder2_path}/*.nii")[0] #os.path.join(folder2_path, os.listdir(folder2_path)[0])
      train_data_dict[folder] = (nii_path, segm_obj_path) #(segm_obj_path, nii_path)
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

    mask = mask.astype(bool)
    
    i = np.any(mask, axis=(1, 2))
    j = np.any(mask, axis=(0, 2))
    k = np.any(mask, axis=(0, 1))

    imin, imax = np.where(i)[0][[0, -1]]
    jmin, jmax = np.where(j)[0][[0, -1]]
    kmin, kmax = np.where(k)[0][[0, -1]]

    # inclusive idxs
    return imin, imax+1, jmin, jmax+1, kmin, kmax+1

def get_bbox_size(imin, imax, jmin, jmax, kmin, kmax):
    return imax - imin, jmax-jmin, kmax-kmin

def print_bbox_size(imin, imax, jmin, jmax, kmin, kmax):
    print(f"{imax - imin}, {jmax-jmin}, {kmax-kmin}")
    
def print_bbox(imin, imax, jmin, jmax, kmin, kmax):
    print(f"Bbox coords: ({imin}, {jmin}, {kmin}) to ({imax}, {jmax}, {kmax}). Size: {imax - imin}, {jmax-jmin}, {kmax-kmin}.")
    print(f"Bounding box coord: from location ({jmin}, {kmin}) of slice {imin} to location ({jmax}, {kmax}) of slice {imax}.")
    #print(f"Slices: {imin}, {imax} ({imax-imin}), Rows: {jmin}, {jmax} ({jmax-jmin}), Cols: {kmin}, {kmax} ({kmax-kmin}).")
        

def get_bbox_vals(i,j,k):
    imin, imax = torch.where(i)[0][[0, -1]]
    jmin, jmax = torch.where(j)[0][[0, -1]]
    kmin, kmax = torch.where(k)[0][[0, -1]]
    
    # inclusive indices
    return torch.tensor([imin, imax+1, jmin, jmax+1, kmin, kmax+1])

def torch_mask2bbox(mask):
    k = torch.any(torch.any(mask, dim=0), dim=0) # 0 -> 1,2 -> 1 -> 2 left
    j = torch.any(torch.any(mask, dim=0), dim=1) # 0 -> 1,2 -> 2 -> 1 left
    i = torch.any(torch.any(mask, dim=1), dim=1) # 1 -> 0,2 -> 0 -> 0 left
    return get_bbox_vals(i,j,k)
        
def batch_get_bbox(yb, preds=False):
    # BCDHW => BDHW, assumed C
    if preds:
        masks = torch.argmax(yb, dim=1).byte()
    else:
        masks =yb.squeeze(1).byte()
    
    # batchwise BDHW
    # BDHW -> BHW -> BW
    # BDHW -> BHW -> BH
    # BDHW -> BDW -> BD 
    bk = torch.any(torch.any(masks, dim=1), dim=1) # 0 -> 1,2 -> 1 -> 2 left
    bj = torch.any(torch.any(masks, dim=1), dim=2) # 0 -> 1,2 -> 2 -> 1 left
    bi = torch.any(torch.any(masks, dim=2), dim=2) # 1 -> 0,2 -> 0 -> 0 left

    # for b in batch
    return torch.stack([get_bbox_vals(i,j,k) for i,j,k in zip(bi,bj,bk)], dim=0)

# masks = np.asarray(torch.argmax(predictions.cpu(), dim=0), dtype=bool)

# def get_bbox_vals(i,j,k):
#     imin, imax = np.where(i)[0][[0, -1]]
#     jmin, jmax = np.where(j)[0][[0, -1]]
#     kmin, kmax = np.where(k)[0][[0, -1]]
    
#     # inclusive indices
#     return imin, imax+1, jmin, jmax+1, kmin, kmax+1
    
# def batch_preds_get_bbox(preds):
#     # BCDHW => BDHW
#     masks = np.asarray(torch.argmax(preds, dim=1).cpu(), dtype=bool)
    
#     # batchwise
#     bi = np.any(masks, axis=(2, 3))
#     bj = np.any(masks, axis=(1, 3))
#     bk = np.any(masks, axis=(1, 2))
    
#     # for b in batch
#     return [get_bbox_vals(i,j,k) for i,j,k in zip(bi,bj,bk)]


# Crop https://github.com/SimpleITK/ISBI2018_TUTORIAL/blob/master/python/03_data_augmentation.ipynb
def threshold_based_crop(image, mask):
    '''
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.  
    '''
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return (sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)]), \
            sitk.RegionOfInterest(mask, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)]))

# get standard reference domain

# src: https://github.com/SimpleITK/ISBI2018_TUTORIAL/blob/master/python/03_data_augmentation.ipynb
def get_reference_frame(objs):
    img_data = [(o.GetSize(), o.GetSpacing()) for o,_ in objs]

    dimension = 3 # 3D MRs
    pixel_id = 2 # 16-bit signed integer

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    for img_sz, img_spc in img_data:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx else mx \
                                      for sz, spc, mx in zip(img_sz, img_spc, reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()


    # Isotropic (1,1,1) pixels
    reference_spacing = np.ones(dimension)
    reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    # Set reference image attributes
    reference_image = sitk.Image(reference_size, pixel_id)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    return reference_image, (reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center)

def get_reference_image(reference_frame):
    reference_size, pixel_id, reference_origin, reference_spacing, reference_direction, reference_center = reference_frame
    reference_image = sitk.Image(reference_size, pixel_id)
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    return reference_image, reference_center

def resample2reference(img, mask, reference_image, reference_center, interpolator = sitk.sitkLinear, default_intensity_value = 0.0, dimension=3):
    
    # Define translation transform mapping origins from reference_image to the current img
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_image.GetOrigin())
    
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    
    return (sitk.Resample(o, reference_image, centered_transform, o_interp, default_intensity_value, o.GetPixelID()) \
                for o, o_interp in ((img, interpolator), (mask, sitk.sitkNearestNeighbor)))
                         
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
    
    return iso_obj, iso_mask_objs