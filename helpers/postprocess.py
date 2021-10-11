# load data
import os, glob

# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk
import torch 

from .general import torch2sitk, sitk2torch

# source sitk 36_Microscopy_Colocalization_Distance_Analysis.html
def get_largest_connected_component(binary_seg):
    # tensor to sitk
    #binary_seg = sitk.GetImageFromArray(binary_seg)
    
    # connected components in sitkSeg
    labeled_seg = sitk.ConnectedComponent(binary_seg)

    # re-order labels according to size (at least 1_000 pixels = 10x10x10)
    labeled_seg = sitk.RelabelComponent(labeled_seg, minimumObjectSize=1000, sortByObjectSize=True)

    # return segm of largest label
    binary_seg = labeled_seg == 1
    
    return binary_seg
    # sitk to tensor
    #return torch.tensor(sitk.GetArrayFromImage(binary_seg))
    
# eval metrics
# evaluate
filters = [sitk.LabelOverlapMeasuresImageFilter(), sitk.HausdorffDistanceImageFilter()]
methods = [
    [
        sitk.LabelOverlapMeasuresImageFilter.GetDiceCoefficient, 
        sitk.LabelOverlapMeasuresImageFilter.GetFalseNegativeError, 
        sitk.LabelOverlapMeasuresImageFilter.GetFalsePositiveError,
        sitk.LabelOverlapMeasuresImageFilter.GetJaccardCoefficient
    ],
    [sitk.HausdorffDistanceImageFilter.GetHausdorffDistance]
]

names = [
    ["dice", "false_neg", "false_pos", "IoU"],
    ["hausdorff_dist"]
]

# d{"dice": x, "Hausdorff": y, "false pos": z}
def eval_measure(ground_truth, after_registration, names_todo=None):
    if isinstance(names_todo, str): names_todo = [names_todo]
        
    d = {}
    for f,method_list, name_list in zip(filters, methods, names):
        for m,n in zip(method_list, name_list):
            if not names_todo or n in names_todo:
                try:
                    #f.Execute(ground_truth, after_registration)
                    f.Execute(after_registration, ground_truth)
                    val = m(f)
                    
                except Exception as e:
                    print(e)
                    val = np.NaN
                d[n] = val
    return d

# both pre and post lcc
def eval_lcc(label, pred, names_todo=None):
    label = torch2sitk(label.squeeze(0).byte())
    pred  = torch2sitk(pred.argmax(0).byte())
    pred_lcc = get_largest_connected_component(pred)
    return eval_measure(label, pred, names_todo), eval_measure(label, pred_lcc, names_todo)

# def eval_pred(label, pred, names_todo=None):
#     label = torch2sitk(label.squeeze(0).byte())
#     pred  = torch2sitk(pred.argmax(0).byte())
#     return eval_measure(label, pred, names_todo)