# Transforms

# 1. Isotropic 3mm or Resize to 50x50x50 dimensions
# 2. Crop/Pad to common dimensions

# - Thanks to faimed3d

from pathlib import Path

import SimpleITK as sitk
import meshio

from fastai.basics import *

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
        im = torch.swapaxes(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
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