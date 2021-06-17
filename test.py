import sys
sys.path.append('/gpfs/home/gologr01/DeepPit')
sys.path.append('/gpfs/home/gologr01/OBELISK')

# imports
# from transforms import AddChannel, Iso, Pad

# Utilities
import os
import time
import pickle
from pathlib import Path

# Fastai
from fastai import *
from fastai.torch_basics import *
from fastai.basics import *

# PyTorch
from torchvision.models.video import r3d_18
from fastai.callback.all import SaveModelCallback
from torch import nn

# 3D extension to FastAI
# from faimed3d.all import *

# Input IO
import SimpleITK as sitk
import meshio

# Numpy and Pandas
import numpy as np
import pandas as pd
from pandas import DataFrame as DF

# Helper functions
from helpers.preprocess import get_data_dict, paths2objs, folder2objs, seg2mask, mask2bbox, print_bbox, get_bbox_size, print_bbox_size
from helpers.general import sitk2np, np2sitk, print_sitk_info, round_tuple, lrange, lmap, get_roi_range, numbers2groups
from helpers.viz import viz_axis

print("hi")