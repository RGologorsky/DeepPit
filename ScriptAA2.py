# imports

# save results
import json 

import os, sys
import numpy as np
import SimpleITK as sitk

from helpers_general import sitk2np, np2sitk, round_tuple, lrange, get_roi_range
from helpers_preprocess import mask2bbox, print_bbox, get_data_dict, folder2objs
from helpers_metrics import compute_dice_coefficient, compute_coverage_coefficient
from helpers_viz import viz_axis


def affine_align(fixed_obj, fixed_mask_obj, moving_obj, moving_mask_obj, param_file = "AffineParamFile.txt"):
    
    # map moving => fixed (the transform is fixed => moving)
    #parameterMap = sitk.GetDefaultParameterMap('affine')
    parameterMap  = sitk.ReadParameterFile(param_file)
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_obj)
    elastixImageFilter.SetMovingImage(moving_obj)
    
    # focus on registering moving mask ROI
    #elastixImageFilter.SetMovingMask(moving_mask_obj)
    #parameterMap["ImageSampler"] = ["RandomSparseMask"]
   
    # print param map
    # sitk.PrintParameterMap(parameterMap)
    
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()

    transformed_moving_obj  = elastixImageFilter.GetResultImage()
    transformedParameterMap = elastixImageFilter.GetTransformParameterMap()[0]
    
    # Binary mask => nearest neighbor
    transformedParameterMap["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    
    # map ROI of moving => fixed
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformedParameterMap)
    transformixImageFilter.SetMovingImage(moving_mask_obj)
    transformixImageFilter.Execute()
    
    transformed_moving_mask_obj = transformixImageFilter.GetResultImage()

    # evaluate: dice, coverage
    fixed_mask_arr              = sitk2np(fixed_mask_obj).astype(bool)
    transformed_moving_mask_arr = sitk2np(transformed_moving_mask_obj).astype(bool)

    dice     = compute_dice_coefficient(fixed_mask_arr, transformed_moving_mask_arr)
    coverage = compute_coverage_coefficient(fixed_mask_arr, transformed_moving_mask_arr)
    
    # save bounding box coords
    bbox_coords = mask2bbox(transformed_moving_mask_arr)
        
    return dice, coverage, bbox_coords, transformed_moving_obj, transformed_moving_mask_arr
    
# save bounding box coords, dice score, generated mask for each (segm obj, nii file) in train_data folder
if __name__ == "__main__":
    results = {}
    
    # load data

    PROJ_PATH = "."

    # Folders containing MR train data
    train_path = f"{PROJ_PATH}/train_data/train_data"
    train_data_dict = get_data_dict(train_path)

    # isotropic preprocessing param
    iso_spacing      = (1,1,1)
    iso_interpolator = sitk.sitkLinear # sitk.sitkBSline
    
    # set fixed MR
    fixed_folder = "50456"
    fixed_obj, fixed_mask_obj = folder2objs(fixed_folder, train_data_dict)
    
    # align all other MRs to fixed
    train_folders = os.listdir(train_path)
    
    #count = 0
    for folder in train_folders:
        if folder != fixed_folder:
            # set moving MR
            moving_obj, moving_mask_obj = folder2objs(folder, train_data_dict)
            dice, coverage, bbox_coords, _, _ = affine_align(moving_obj, moving_mask_obj, fixed_obj, fixed_mask_obj)
            #affine_align(fixed_obj, fixed_mask_obj, moving_obj, moving_mask_obj)
            
            # save results
            results["fixed"] = fixed_folder
            results[folder] = {"dice": float(dice), "coverage": float(coverage), "bbox_coords": tuple(int(c) for c in bbox_coords)}
            
            # print
            print(f"Folder {folder}. Results: {results[folder]}.")
            
            # break
            #count += 1
            #if count == 3:
            #    break
                
    # write results
    with open("resultsVaryingFixedObj.json", 'w') as outfile:
        json.dump(results, outfile)