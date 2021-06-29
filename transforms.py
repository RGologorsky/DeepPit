# Transforms

# 1. Isotropic 3mm or Resize to 50x50x50 dimensions
# 2. Crop/Pad to common dimensions
# 3. Nyul-Udupa Piecewise Hist w/ max scaling
# 4. MattAffineTfm

# - Thanks to faimed3d

from pathlib import Path

import SimpleITK as sitk
import meshio

from fastai.basics import *

from helpers.nyul_udupa import piecewise_hist

class Hi(Transform): pass

class AddChannel(DisplayedTransform):
    "Adds Channels dims, in case they went missing"
    split_idx,order = None, 99
        
    def encodes(self, x:Tensor):
        if x.ndim == 3: x = x.unsqueeze(0)
        if x.ndim == 4: x = x.unsqueeze(1)
        return x
    
class Iso(ItemTransform):
    
    def __init__(self, new_sp = 3):
        self.new_sp = new_sp
        
    def encodes(self, x):
        # get sitk objs
        im_path, segm_path = x
        mr = sitk.ReadImage(im_path, sitk.sitkFloat32)
        im = torch.transpose(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
        # im = torch.swapaxes(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
        mk = torch.load(f"{str(Path(segm_path).parent)}/seg.pt").float()

        # resize so isotropic spacing
        orig_sp = mr.GetSpacing()
        orig_sz = mr.GetSize()
        new_sz = [int(round(osz*ospc/self.new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

        while im.ndim < 5: 
            im = im.unsqueeze(0)
            mk = mk.unsqueeze(0)

        return F.interpolate(im, size = new_sz, mode = 'trilinear', align_corners=False).squeeze(), \
                F.interpolate(mk, size = new_sz, mode = 'nearest').squeeze().long()    

# pad to new size
class PadSz(Transform):
    def __init__(self, new_sz):
        self.new_sz = new_sz
    
    def encodes(self, arr):
        pad = [x-y for x,y in zip(self.new_sz, arr.shape)]
        pad = [a for amt in pad for a in (amt//2, amt-amt//2)]
        pad.reverse()
        
        return F.pad(arr, pad, mode='constant', value=0) 


# NYUL-UDUPA, from helpers.nyul_udapa import piecewise hist

# Transform: nyul udupa hist norm, max scale
@patch
def max_scale(t:torch.Tensor):
    t = (t - t.min()) / (t.max() - t.min())
    return t

class PiecewiseHistScaling(ItemTransform):
    """
    Applies theNyul and Udupa histogram nomalization and rescales the pixel values.

    Args:
        input_image (TensorDicom3D): image on which to find landmarks
        landmark_percs (torch.tensor): corresponding landmark points of standard scale
        final_scale (function): final rescaling of values, if none is provided values are
                                scaled to a mean of 0 and a std of 1.
        slicewise (bool): if the scaling should be applied to each slice individually. Slower but leads to more homogeneous images.

    Returns:
        If input is TensorMask3D returns input unchanged
        If input is TensorDicom3D returns normalized and rescaled Tensor

    """
    def __init__(self, landmark_percs=None, standard_scale=None, final_scale=None):
        self.landmark_percs = landmark_percs
        self.standard_scale = standard_scale
        self.final_scale = final_scale

    def encodes(self, item):
        x,mk = item
        x = x.piecewise_hist(self.landmark_percs, self.standard_scale)
        x = x.clamp(min=0)
        x = x.sqrt().max_scale() if self.final_scale is None else self.final_scale(x)
        #return x
        return x, mk
        
# from OBELISK mattiaspaul
class MattAffineTfm(ItemTransform):
    # split_idx: 0 for train, 1 for validation, None for both
    split_idx,order = 0, 100
    
    def __init__(self, strength=0.05):
        self.strength = strength
        
    # batches
    def encodes(self, batch_items):
        img_in, seg_in = batch_items
        B,C,D,H,W = img_in.size()
        affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * self.strength).to(img_in.device)
        
        meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))

        img_out = F.grid_sample(img_in, meshgrid,padding_mode='border')
        seg_out = F.grid_sample(seg_in.float(), meshgrid, mode='nearest').long()

        return img_out, seg_out
        
        
# OLD

# # crop center
# class CenterCropTfm(Transform):
#     def __init__(self, size):
#         self.size = size
        
#     def encodes(self, arr):
#         return self.cropND(arr, self.size)
    
#     # https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
#     @staticmethod
#     def cropND(img, bounding):
#         start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
#         end = tuple(map(operator.add, start, bounding))
#         slices = tuple(map(slice, start, end))
#         return img[slices]
    
# # crop by coords
# class CropBBox(Transform):
#     def __init__(self, bbox):
#         self.bbox = bbox
    
#     def encodes(self, arr):
#         imin, imax, jmin, jmax, kmin, kmax = self.bbox
#         cropped = arr[imin:imax, jmin:jmax, kmin:kmax]
        
#         # pad if needed
#         new_size = [imax-imin, jmax-jmin, kmax-kmin]
        
#         pad = [x-y for x,y in zip(new_size, arr.shape)]
#         pad = [a for amt in pad for a in (amt//2, amt-amt//2)]
#         pad.reverse()
        
#         return F.pad(arr, pad, mode='constant', value=0)
    
    
# class OLD_DoAll(ItemTransform):
    
#     def __init__(self, new_sp = 3):
#         self.new_sp = new_sp
        
#     def encodes(self, x):
#         # get sitk objs
#         im_path, segm_path = x
#         folder  = Path(segm_path).parent.name
#         ras_adj = int(folder) in range(50455, 50464)

#         mr         = sitk.ReadImage(im_path, sitk.sitkFloat32)
#         segm       = meshio.read(segm_path)
#         mask_arr   = seg2mask(mr, segm, ras_adj)

#         # resize so isotropic spacing
#         orig_sp = mr.GetSpacing()
#         orig_sz = mr.GetSize()
#         new_sz = [int(round(osz*ospc/self.new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

#         im = torch.swapaxes(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
#         mk = torch.tensor(mask_arr).float()

#         while im.ndim < 5: 
#             im = im.unsqueeze(0)
#             mk = mk.unsqueeze(0)

#         return F.interpolate(im, size = new_sz, mode = 'trilinear', align_corners=False).squeeze(), \
#                 F.interpolate(mk, size = new_sz, mode = 'nearest').squeeze().long()