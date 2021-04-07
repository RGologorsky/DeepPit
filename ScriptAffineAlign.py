# imports
import os, sys

# save results
import json

# filter filenames for .nii
import glob

# numpy
import numpy as np

# simple itk for dicom
import SimpleITK as sitk

# meshio for 3DSlicer segm obj
import meshio

# helpers: segmentation,np2sitk conversion, misc
from helpers import seg2mask, get_roi_range, \
                    sitk2np, np2sitk, \
                    get_isotropic, \
                    round_tuple, lrange, \
                    compute_dice_coefficient, compute_coverage, \
                    bbox, print_bbox


#######################

# load data

PROJ_PATH = "."

# Load train data
train_path = f"{PROJ_PATH}/train_data/train_data"

# Folders containing MR train data
train_folders = os.listdir(train_path)
print(f"Train data folders: {train_folders}")

# make a dictionary of key = train folder, value = (segm obj, nii file)
train_data_dict = {}
for folder in train_folders:
  segm_obj_path = os.path.join(train_path, folder, "Segmentation.obj")

  mp_path      = os.path.join(train_path, folder, "MP-RAGE")
  folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
  folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])
  nii_path     = glob.glob(f"{folder2_path}/*.nii")[0] #os.path.join(folder2_path, os.listdir(folder2_path)[0])
  train_data_dict[folder] = (segm_obj_path, nii_path)

#############################

# param
iso_spacing      = (1,1,1)
iso_interpolator = sitk.sitkLinear # sitk.sitkBSline

##############################

# given folder name, return isotropic SITK obj of nii and segm obj
def folder2objs(folder_name):
    segm_path, file = train_data_dict[folder_name]

    # compile MR obj from nii file using Simple ITK reader
    obj        = sitk.ReadImage(file)
    segm       = meshio.read(segm_path)
    mask_arr   = seg2mask(obj, segm)
    
    # preprocess
    
    # 1. isotropic
    iso_obj       = get_isotropic(obj, iso_spacing, iso_interpolator)
    iso_mask_obj  = get_isotropic(np2sitk(mask_arr, obj), iso_spacing, iso_interpolator)
    
    return iso_obj, iso_mask_obj
    
def affine_align(fixed_obj, fixed_mask_obj, moving_obj, moving_mask_obj):
    
    # map moving => fixed (the transform is fixed => moving)
    parameterMap = sitk.GetDefaultParameterMap('affine')

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_obj)
    elastixImageFilter.SetMovingImage(moving_obj)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()

    transformed_moving_obj  = elastixImageFilter.GetResultImage()
    transformedParameterMap = elastixImageFilter.GetTransformParameterMap()
    
    # map ROI of moving => fixed
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformedParameterMap)
    transformixImageFilter.SetMovingImage(moving_mask_obj)
    transformixImageFilter.Execute()
    
    transformed_moving_mask_obj = transformixImageFilter.GetResultImage()

    # save bounding box coords
    bbox_coords = bbox(transformed_moving_mask_obj)
    
    # evaluate: dice, coverage
    fixed_mask_arr              = sitk2np(fixed_mask_obj).astype(bool)
    transformed_moving_mask_arr = sitk2np(transformed_moving_mask_obj).astype(bool)
    
    dice     = compute_dice_coefficient(fixed_mask_arr, transformed_moving_mask_arr)
    coverage = compute_coverage(fixed_mask_arr, transformed_moving_mask_arr)
    
    return dice, coverage, bbox_coords

# save bounding box coords, dice score, generated mask for each (segm obj, nii file) in train_data folder
if __name__ == "__main__":
    results = {}
    
    # set fixed MR
    fixed_folder = "50456"
    
    print(f"Getting fixed obj from folder {fixed_folder}")
    fixed_obj, fixed_mask_obj = folder2objs(fixed_folder)
    
    # align all other MRs to fixed
    for folder in train_folders:
        print(f"Getting aligment with obj from folder {folder}")
        if folder != fixed_folder:
            # set moving MR
            moving_obj, moving_mask_obj = folder2objs(folder)
            dice, coverage, bbox_coords = affine_align(fixed_obj, fixed_mask_obj, moving_obj, moving_mask_obj)
            
            # save results
            results["fixed"] = fixed_folder
            results[folder] = {"dice": dice, "coverage": coverage, "bbox_coords": bbox_coords}
            
            print(f"Results: dice: {dice}, coverage: {coverage}, bbox_coords: {bbox_coords}")
            
    # write results
    with open("results.json", 'wb') as outfile:
        json.dump(results, outfile)
    