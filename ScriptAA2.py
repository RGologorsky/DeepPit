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

# old code
# focus on registering moving mask ROI
#elastixImageFilter.SetMovingMask(moving_mask_obj)
#parameterMap["ImageSampler"] = ["RandomSparseMask"]

# print param map
# sitk.PrintParameterMap(parameterMap)

#transformed_moving_obj  = elastixImageFilter.GetResultImage()

def align_and_tfm(fixed_obj, moving_obj, moving_mask_obj, param_folder = "ElastixParamFiles", param_files = ["affine.txt", "bspline.txt"]):
    
    # ALIGN ATLAS AND INPUT IMAGE
    
    # set moving and fixed images (resample moving=>fixed using T:fixed=>moving)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_obj)
    elastixImageFilter.SetMovingImage(moving_obj)
    
    # set parameter map
    parameterMapVector = sitk.VectorOfParameterMap()
    for param_file in param_files:
        parameterMapVector.append(sitk.ReadParameterFile(f"{param_folder}/{param_file}"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    # Execute alignment
    elastixImageFilter.Execute()

    # MAP MOVING (ATLAS BINARY ROI) ONTO FIXED (INPUT) 

    # set moving image (atlas)                                                    
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(moving_mask_obj)
                    
    # set parameter map (Binary mask => nearest neighbor final interpolation)
    transformedParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    transformedParameterMapVector[-1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transformixImageFilter.SetTransformParameterMap(transformedParameterMapVector)

    # Execute transformation
    transformixImageFilter.Execute()
    
    return transformixImageFilter.GetResultImage()

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
    
    # set atlas MR
    atlas_folder = "50456"
    atlas_obj, atlas_mask_obj = folder2objs(atlas_folder, train_data_dict)
    
    # align all other MRs to fixed
    train_folders = [f for f in os.listdir(train_path) if f != atlas_folder]
    
    count = 0
    for folder in train_folders:
        # set input MR
        input_obj, gt_mask_obj = folder2objs(folder, train_data_dict)

        # Align the atlas and the input MR. Resample atlas ROI onto input ROI (fixed: input, moving: atlas). 
        pred_mask_obj = align_and_tfm(input_obj, atlas_obj, atlas_mask_obj, \
                                      param_folder = "ElastixParamFiles", param_files = ["affine.txt", "bspline.txt"])

        # Evaluate predicted input ROI
        gt_mask_arr   = sitk2np(gt_mask_obj).astype(bool)
        pred_mask_arr = sitk2np(pred_mask_obj).astype(bool)

        dice     = compute_dice_coefficient(gt_mask_arr, pred_mask_arr)
        coverage = compute_coverage_coefficient(gt_mask_arr, pred_mask_arr)
        bbox_coords = mask2bbox(pred_mask_arr)

        # save results
        results["fixed"] = fixed_folder
        results[folder] = {"dice": float(dice), "coverage": float(coverage), "bbox_coords": tuple(int(c) for c in bbox_coords)}

        # print
        print(f"Folder {folder}. Results: {results[folder]}.")

        # break
        count += 1
        if count == 3:
           break
                
    # write results
    with open("resultsVaryingFixedObj.json", 'w') as outfile:
        json.dump(results, outfile)