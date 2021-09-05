# General Helper fns
#
#  1. sitk2np / np2sitk
#  2. round_tuple
#  3. lrange
#  4. get_roi_range
#  5. numbers2groups (pretty print consec numbers as groups) 
#  6. print_hardware_stats
#  7. rm_prefix (removes prefix from each item in list, optional skip)
#  8. get_param

# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk

# model name
from pathlib import Path

# consecutive numbers
from operator import itemgetter
from itertools import groupby

# hardware stats
import os
import torch
import GPUtil as GPU

def print_hardware_stats():
    # print GPU/CPU counts
    gpu_count = torch.cuda.device_count()
    cpu_count = os.cpu_count()
    print("#GPU = {0:d}, #CPU = {1:d}".format(gpu_count, cpu_count))

    # print GPU stats
    GPUs = GPU.getGPUs()
    for gpu in GPUs:
        print("GPU {0:20s} RAM Free: {1:.0f}MB | Used: {2:.0f}MB | Util {3:3.0f}% | Total {4:.0f}MB".format(gpu.name, gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

# sitk obj and np array have different index conventions
# deep copy
def sitk2np(obj): return np.swapaxes(sitk.GetArrayFromImage(obj), 0, 2)
def np2sitk(arr): return sitk.GetImageFromArray(np.swapaxes(arr, 0, 2))

def torch2sitk(t): return sitk.GetImageFromArray(torch.transpose(t, 0, 2))
def sitk2torch(o): return torch.transpose(torch.tensor(sitk.GetArrayFromImage(o)), 0, 2)

# numpy mask arr into sitk obj
def mask2sitk(mask_arr, sitk_image):
  # convert bool mask to int mask
  # swap axes for sitk
  obj = sitk.GetImageFromArray(np.swapaxes(mask_arr.astype(np.uint8), 0, 2))
  obj.SetOrigin(sitk_image.GetOrigin())
  obj.SetSpacing(sitk_image.GetSpacing())   
  obj.SetDirection(sitk_image.GetDirection())
  return obj

def print_sitk_info(image):    
    print("Size: ", image.GetSize())
    print("Origin: ", image.GetOrigin())
    print("Spacing: ", image.GetSpacing())
    print("Direction: ", image.GetDirection())
    print(f"Pixel type: {image.GetPixelIDValue()} = {image.GetPixelIDTypeAsString()}")

# round all floats in a tuple to 3 decimal places
def round_tuple(t, d=3): return tuple(round(x,d) for x in t)

# returns range obj as list
def lrange(a,b): return list(range(a,b))

# applies function to list of values
def lmap(fn, arr, unpack_input=False, unpack_output=False):
    output = [fn(*o) for o in arr] if unpack_input else [fn(o) for o in arr]
    return zip(*output) if unpack_output else output

# see which slices contain ROI
def get_roi_range(bin_mask_arr, axis):
  slices = np.unique(np.nonzero(bin_mask_arr)[axis])
  return min(slices), max(slices)

# find groups of consecutive numbers (e.g. [1,2,3,5,6,7] returnns (1,3), (5,7)
# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def numbers2groups(data):
    ranges = []
    for key, group in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), group))
        if len(group) > 1:
            ranges.append(range(group[0], group[-1]))
        else:
            ranges.append(group[0])
    return ranges

def cut(string, prefix):
    return string[len(prefix)+1:]

def rm_prefix(lst, prefix, skip=1, do_sort=False):
    if isinstance(lst, str): lst = [lst]
    lst = [cut(lst[i], prefix) for i in range(0,len(lst),skip)]
    return sorted(lst) if do_sort else lst

def rm_prefix_item(lst, prefix, skip=1, do_sort=False):
    if isinstance(lst, str): lst = [lst]
    lst = [(cut(lst[i][0], prefix[0]), cut(lst[i][1], prefix[1])) for i in range(0,len(lst),skip)]
    return sorted(lst) if do_sort else lst

def get_param(fn, prefix, suffix):
    start = fn.index(prefix)
    end   = fn.index(suffix)
    res   = fn[start+len(prefix):end].split("_")

    # single result
    if len(res) == 1: 
        try:
            return int(res[0])
        except:
            return res[0]

    # multiple results - list of ints or str
    try:
        return [int(x) for x in res]
    except:
        return "_".join(res)
    
# params
def get_param_default(name, prefix, suffix, default):
    try:
        return get_param(name, prefix, suffix)
    except:
        return default
    
# get params from model name
def modelfn2dict(fn):
    model_name = Path(fn).name
    
    # get params from model name
    model_type = get_param(model_name, "model_", "_loss")

    if "loss_bs" in model_name:
        loss_type  = get_param(model_name, "loss_", "_bs")
    else:
        loss_type  = get_param(model_name, "loss_", "_full_res")
    
    full_res   = get_param_default(model_name, "full_res_", "_pixdim", 96)
    pixdim     = get_param_default(model_name, "pixdim_", "_do_simple", 1.5)
    do_simple  = get_param_default(model_name, "do_simple_", "_do_flip", False)
    do_flip    = get_param_default(model_name, "do_flip_", "_bs", True)

    # tuple
    pixdim    = tuple(float(pixdim) for _ in range(3))
    full_res  = tuple(int(full_res) for _ in range(3))

    # bool
    do_flip   = do_flip == "True"
    do_simple = do_simple == "True"
    
    return {"model_type": model_type, "loss_type": loss_type, \
            "full_res":full_res, "pixdim":pixdim, \
            "do_simple":do_simple, "do_flip":do_flip}