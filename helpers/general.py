# General Helper fns
#
#  1. sitk2np / np2sitk
#  2. round_tuple
#  3. lrange
#  4. get_roi_range
#  5. numbers2groups (pretty print consec numbers as groups) 

# numpy to SITK conversion
import numpy     as np
import SimpleITK as sitk

# consecutive numbers
from operator import itemgetter
from itertools import groupby

# sitk obj and np array have different index conventions
# deep copy
def sitk2np(obj): return np.swapaxes(sitk.GetArrayFromImage(obj), 0, 2)
def np2sitk(arr): return sitk.GetImageFromArray(np.swapaxes(arr, 0, 2))

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
