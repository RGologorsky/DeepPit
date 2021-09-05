# Transforms

# 1. Isotropic 3mm or Resize to 50x50x50 dimensions
# 2. Crop/Pad to common dimensions
# 3. Nyul-Udupa Piecewise Hist w/ max scaling
# 4. MattAffineTfm

# - Thanks to faimed3d

from pathlib import Path

import SimpleITK as sitk
import meshio

# Displayed and Rand Transform, decorator @patch
from fastai.basics import *
from fastai import *

from fastai.vision.augment import RandTransform

import torchvision as tv
import torchvision.transforms.functional as _F

from helpers.nyul_udupa import piecewise_hist
    
class UndoDict(ItemTransform):
    split_idx = None

    def __init__(self, image_key = "image", label_key = "label"):
        self.image_key = image_key
        self.label_key = label_key
        
    def encodes(self, d):
        return d[self.image_key], d[self.label_key]
    
    def __str__(self):
        return f"UndoDict"
    
class Path2Tensor(Transform):
    def encodes(self, fn):
        mr = sitk.ReadImage(fn, sitk.sitkFloat32)
        im = torch.transpose(torch.tensor(sitk.GetArrayFromImage(mr)), 0, 2)
        return im
    
class AddChannel(DisplayedTransform):
    "Adds Channels dims, in case they went missing"
    split_idx,order = None, 99
        
    def encodes(self, x:Tensor):
        if x.ndim == 3: x = x.unsqueeze(0)
        if x.ndim == 4: x = x.unsqueeze(1)
        return x
    
class Iso(ItemTransform):
    split_idx = None
    
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

    def __str__(self):
        return f"Iso(new_sp = {self.new_sp})"
        
# pad to new size
class PadSz(Transform):
    split_idx = None
    
    def __init__(self, new_sz):
        self.new_sz = new_sz
    
    def encodes(self, arr:torch.Tensor):
        pad = [x-y for x,y in zip(self.new_sz, arr.shape)]
        pad = [a for amt in pad for a in (amt//2, amt-amt//2)]
        pad.reverse()
        
        return F.pad(arr, pad, mode='constant', value=0) 
    
    def __str__(self):
        return f"PadSz(new_sz = {self.new_sz})"


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
def augmentAffine(img_in, seg_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B,C,D,H,W = img_in.size()
    affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))

    img_out = F.grid_sample(img_in, meshgrid,padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)

    return img_out, seg_out

class MattAffine(ItemTransform):
    split_idx = 0
    order = 100
 
    def __init__(self, p=0.5, strength=0.05):
        self.strength = strength
        self.do = (p==1.) or random.random() < p
        
    def encodes(self, x):  
        if not self.do: return x
        
        im,mk = x
        
        B,C,D,H,W = im.size()
        affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * self.strength).to(im.device)
        
        meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))

        im = F.grid_sample(im, meshgrid,padding_mode='border')
        mk = F.grid_sample(mk.float(), meshgrid, mode='nearest').long()

        return im, mk
        
class ZScale(ItemTransform):
    split_idx = None
    
    """
    normalize a target image by subtracting the mean of foreground pixels
    and dividing by the standard deviation
    Source: https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/zscore.py
    """
 
    # batch BCDHW
    def encodes(self, item):
        im,mk = item
        
        # non-zero
        mask = im > im.mean()
        
        # rescale
        mean = im[mask].mean()
        std  = im[mask].std()
        
        return (im-mean)/std, mk

# https://fastai1.fast.ai/vision.image.html#RandTransform
# https://kbressem.github.io/faimed3d/transforms.html
def gaussian_noise(t, std):
    noise = torch.randn(t.shape).to(t.device)
    return t + (std**0.5)*noise*t

class GNoise(ItemTransform):
    split_idx = 0
    
    def __init__(self, p=0.5, std_range=[0.01, 0.1]):
        self.p = p
        self.lwr_std = np.min(std_range)
        self.upr_std = np.max(std_range)

    def encodes(self, x):
        do = self.p==1. or random.random() < self.p
        if not do: return x
        
        self.std = random.choice(np.arange(self.lwr_std,
                                           self.upr_std,
                                           self.lwr_std))
        im,mk = x
        return gaussian_noise(im, self.std), mk

class GBlur(ItemTransform):
    split_idx = 0
    
    def __init__(self, p=0.5, kernel_size_range=[5, 11], sigma=0.5):
        self.p = p
        self.kernel_size_range = kernel_size_range
        self.sigma = sigma

    def encodes(self, x):
        
        # to be or not to be
        do = self.p==1. or random.random() < self.p
        if not do: return x
        
        # before call
        sizes = range(self.kernel_size_range[0],
                      self.kernel_size_range[1],
                      2)
        self.kernel = random.choices(sizes, k = 2)
        
        # do call
        im, mk = x
        return _F.gaussian_blur(im, self.kernel, self.sigma), mk


class MattAff(ItemTransform):
    split_idx = 0
    order = 100
    
    def __init__(self, p=0.5, strength=0.05):
        self.p        = p
        self.strength = strength
    
    def encodes(self, x):
        do = self.p==1. or random.random() < self.p
        if not do: return x
        
        im, mk = x
                
        B,C,D,H,W = im.size()
        affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * self.strength).to(im.device)
        
        meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))

        im_out = F.grid_sample(im, meshgrid,padding_mode='border')
        mk_out = F.grid_sample(mk.float(), meshgrid, mode='nearest') #.long()

        return im_out, mk_out
    
    def __str__(self):
        return f"MattAff(p = {self.p}, strength = {self.strength})"
    
@patch
def adjust_brightness(x:Tensor, beta):
    return torch.clamp(x + beta, x.min(), x.max())


class RandBright(ItemTransform):
    split_idx = 0
    
    def __init__(self, p=0.5, beta_range=[-0.3, 0.3]):
        self.p = p
        self.lwr_beta = np.min(beta_range)
        self.upr_beta = np.max(beta_range)

    def encodes(self, x):
        do = self.p==1. or random.random() < self.p
        if not do: return x
        
        self.beta = random.choice(np.arange(self.lwr_beta,
                                           self.upr_beta,
                                           25))
        
        im, mk = x
        return im.adjust_brightness(self.beta), mk

# Cell
@patch
def adjust_contrast(x:Tensor, alpha):
    x2 = x*alpha
    min = x2.min() + abs(x2.min() - x.min())
    return torch.clamp(x2, min, x.max())

class RandContrast(ItemTransform):
    split_idx = 0
    
    def __init__(self, p=0.6, alpha_range=[0.7, 2.0]):
        self.p         = p
        self.lwr_alpha = np.min(alpha_range)
        self.upr_alpha = np.max(alpha_range)

    def encodes(self, x):
        do = self.p==1. or random.random() < self.p
        if not do: return x
        
        self.alpha = random.choice(np.arange(self.lwr_alpha,
                                            self.upr_alpha,
                                            25))
        print("a", self.alpha)
        im, mk = x
        return im.adjust_contrast(self.alpha), mk
# Cell

@patch
def dihedral3d(x:Tensor, k):
    "apply dihedral transforamtions to the 3D Dicom Tensor"
    if k in [6,7,8,9,14,15,16,17]: x = x.flip(-3)
    if k in [4,10,11,14,15]: x = x.flip(-1)
    if k in [5,12,13,16,17]: x = x.flip(-2)
    if k in [1,7,10,12,14,16]:
        if x.ndim == 3: x = x.transpose(1, 2)
        else: x = x.transpose(2,3)
    if k in [2,8]: x = x.flip(-1).flip(-2)
    if k in [3,11,13,15,17]:
        if x.ndim == 3: x = x.transpose(1, 2).flip(-1)
        else: x = x.transpose(2,3).flip(-1)
    return x

class RandDihedral(ItemTransform):
    "randomly flip and rotate the 3D Dicom volume with a probability of `p`"
    
    split_idx = 0
    
    def __init__(self, p):
        self.p = p
        
    def encodes(self, x):
        do = self.p==1. or random.random() < self.p
        if not do: return x
        
        self.k = random.randint(0,17)
        
        im,mk = x
        return im.dihedral3d(self.k), mk.dihedral3d(self.k)

# https://github.com/jcreinhold/niftidataset/blob/master/niftidataset/transforms.py
# def _interpolation_modes_from_int(i: int) -> InterpolationMode:
#     inverse_modes_mapping = {
#         0: InterpolationMode.NEAREST,
#         2: InterpolationMode.BILINEAR,
#         3: InterpolationMode.BICUBIC,
#         4: InterpolationMode.BOX,
#         5: InterpolationMode.HAMMING,
#         1: InterpolationMode.LANCZOS,
#     }
#     return inverse_modes_mapping[i]
    
# class RandomAffine(tv.transforms.RandomAffine):
#     """ apply random affine transformations to a sample of images """

#     def __init__(self, p: float, degrees: float, translate: float = 0, scale: float = 0,
#                  interpolation: int = 2): # bilinear interp
#         self.p = p
#         self.degrees, self.translate, self.scale = (-degrees, degrees), (translate, translate), (1 - scale, 1 + scale)
#         self.shear, self.fill = None, 0
#         self.resample = interpolation

#     def affine(self, x, params, interpolation=2):
#         return _F.affine(x, *params, resample=interpolation)

#     def __call__(self, x):
#         im, mk = x
#         ret = self.get_params(self.degrees, self.translate, self.scale, None, mk.size())
#         if self.degrees[1] > 0 and random.random() < self.p:
#             return self.affine(im, ret, self.resample), self.affine(mk, ret, 0)
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
    
    
