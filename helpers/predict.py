# Predict Helper fns
#
#  1. atlas2pred
#


# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk

from .general import sitk2np, np2sitk
from .preprocess import mask2bbox

# https://github.com/SuperElastix/elastix/blob/522843d90ff586be051c480514cd14a88db45dbf/src/Core/Main/elxParameterObject.cxx#L260-L362
def get_parameter_map(transformName = "affine"):
  parameterMap = sitk.ParameterMap()

  # NumberOfResolutions 1 or 4, ResampleInterpolator Final
  
  # from affine.txt - number of resolutions 1 or 4, max number registrations 256 or 500,
        
  # Common Components
  parameterMap[ "FixedImagePyramid" ]              = [ "FixedSmoothingImagePyramid" ];
  parameterMap[ "MovingImagePyramid" ]             = [ "MovingSmoothingImagePyramid" ];
  parameterMap[ "Interpolator" ]                   = [ "LinearInterpolator" ];
  parameterMap[ "Optimizer" ]                      = [ "AdaptiveStochasticGradientDescent" ];
  parameterMap[ "Resampler" ]                      = [ "DefaultResampler" ];
  parameterMap[ "ResampleInterpolator" ]           = [ "FinalBSplineInterpolator" ];
  parameterMap[ "FinalBSplineInterpolationOrder" ] = [ "1" ];
  parameterMap[ "NumberOfResolutions" ]            = [ "1" ];
  parameterMap[ "WriteIterationInfo" ]             = [ "false" ];

  # Image Sampler
  parameterMap[ "ImageSampler" ]                    = [ "RandomCoordinate" ];
  parameterMap[ "NumberOfSpatialSamples" ]          = [ "2048" ];
  parameterMap[ "CheckNumberOfSamples" ]            = [ "true" ];
  parameterMap[ "MaximumNumberOfSamplingAttempts" ] = [ "8" ];
  parameterMap[ "NewSamplesEveryIteration" ]        = [ "true" ];

  # Optimizer
  parameterMap[ "NumberOfSamplesForExactGradient" ] = [ "4096" ];
  parameterMap[ "DefaultPixelValue" ]               = [ "0.0" ];
  parameterMap[ "AutomaticParameterEstimation" ]    = [ "true" ];

  # Output
  parameterMap[ "WriteResultImage" ]  = [ "false" ];
  parameterMap[ "ResultImageFormat" ] = [ "nii" ];

  # Misc
  parameterMap[ "UseDirectionCosines" ]                    = [ "true"]
  parameterMap[ "AutomaticScalesEstimation" ]              = [ "true" ]

  parameterMap[  "AutomaticTransformInitialization" ]      = [ "false" ]
  parameterMap[ "AutomaticTransformInitializationMethod" ] = [ "GeometricalCenter" ]

  parameterMap[ "ShowExactMetricValue" ]        = [ "false" ]
  parameterMap[ "UseMultiThreadingForMetrics" ] = [ "true" ]
  parameterMap[ "UseFastAndLowMemoryVersion" ]  = [ "true" ]

  # transformNames
  if( transformName == "translation" ):
    parameterMap[ "Registration" ]              = [ "MultiResolutionRegistration" ];
    parameterMap[ "Transform" ]                 = [ "TranslationTransform" ];
    parameterMap[ "Metric" ]                    = [ "AdvancedMattesMutualInformation" ];
    parameterMap[ "MaximumNumberOfIterations" ] = [ "256" ];
  
  elif( transformName == "rigid" ):
    parameterMap[ "Registration" ]              = [ "MultiResolutionRegistration" ];
    parameterMap[ "Transform" ]                 = [ "EulerTransform" ];
    parameterMap[ "Metric" ]                    = [ "AdvancedMattesMutualInformation" ];
    parameterMap[ "MaximumNumberOfIterations" ] = [ "256" ];
  
  elif( transformName == "affine" ):
    parameterMap[ "Registration" ]              = [ "MultiResolutionRegistration" ];
    parameterMap[ "Transform" ]                 = [ "AffineTransform" ];
    parameterMap[ "Metric" ]                    = [ "AdvancedMattesMutualInformation" ];
    parameterMap[ "MaximumNumberOfIterations" ] = [ "256" ];

  else:
      pass

  return parameterMap

def atlas2pred(input_arr, atlas_arr, atlas_mask_arr):
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetLogToConsole(False)

    # set parameter map
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(get_parameter_map("affine"))

#     param_folder = "ElastixParamFiles"
#     param_files = ["affine.txt"]
#     for param_file in param_files:
#         parameterMapVector.append(sitk.ReadParameterFile(f"{param_folder}/{param_file}"))

    elastixImageFilter.SetParameterMap(parameterMapVector)

    # set moving and fixed images (resample moving=>fixed using T:fixed=>moving)
    
    # input = fixed, atlas = moving
    elastixImageFilter.SetFixedImage(np2sitk(input_arr))
    elastixImageFilter.SetMovingImage(np2sitk(atlas_arr))
    elastixImageFilter.Execute()

    #pred_obj = elastixImageFilter.GetResultImage()

    # MAP MOVING (ATLAS BINARY ROI) ONTO FIXED (INPUT) 

    # set moving image (atlas)                                                    
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetLogToConsole(False)
    
    transformixImageFilter.SetMovingImage(np2sitk(atlas_mask_arr))

    # set parameter map (Binary mask => nearest neighbor final interpolation)
    transformedParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    transformedParameterMapVector[-1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transformixImageFilter.SetTransformParameterMap(transformedParameterMapVector)

    # Execute transformation
    transformixImageFilter.Execute()
    
    # pred_mask_obj = transformixImageFilter.GetResultImage()
    return sitk2np(transformixImageFilter.GetResultImage()).astype(bool)


# def atlas2pred(input_obj, atlas_obj, atlas_mask_obj):
    
#     elastixImageFilter = sitk.ElastixImageFilter()

#     # set parameter map
#     parameterMapVector = sitk.VectorOfParameterMap()
#     parameterMapVector.append(get_parameter_map("affine"))

#     elastixImageFilter.SetParameterMap(parameterMapVector)

#     # set moving and fixed images (resample moving=>fixed using T:fixed=>moving)
    
#     # input = fixed, atlas = moving
#     elastixImageFilter.SetFixedImage(input_obj)
#     elastixImageFilter.SetMovingImage(atlas_obj)
#     elastixImageFilter.Execute()

#     #pred_obj = elastixImageFilter.GetResultImage()

#     # MAP MOVING (ATLAS BINARY ROI) ONTO FIXED (INPUT) 

#     # set moving image (atlas)                                                    
#     transformixImageFilter = sitk.TransformixImageFilter()
#     transformixImageFilter.SetMovingImage(atlas_mask_obj)

#     # set parameter map (Binary mask => nearest neighbor final interpolation)
#     transformedParameterMapVector = elastixImageFilter.GetTransformParameterMap()
#     transformedParameterMapVector[-1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
#     transformixImageFilter.SetTransformParameterMap(transformedParameterMapVector)

#     # Execute transformation
#     transformixImageFilter.Execute()
    
#     # pred_mask_obj = transformixImageFilter.GetResultImage()
#     return transformixImageFilter.GetResultImage()


# mask2bbox(sitk2np(atlas2pred(input_obj, atlas_obj, atlas_mask_obj)))
def atlas2pred_bbox(input_obj, atlas_obj, atlas_mask_obj):
    pred_mask_obj = atlas2pred(input_obj, atlas_obj, atlas_mask_obj)
    bbox = mask2bbox(sitk2np(pred_mask_obj))
    del pred_mask_obj
    return bbox


# old
    #param_folder = "ElastixParamFiles"
    #param_files = ["affine.txt"]
#     for param_file in param_files:
#         parameterMapVector.append(sitk.ReadParameterFile(f"{param_folder}/{param_file}"))
    