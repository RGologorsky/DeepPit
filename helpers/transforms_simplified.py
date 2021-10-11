import torch
from pprint import pprint, pformat

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPad,
    SpatialPadd,
    RandRotate90d,
    ToTensord,
)

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)

from fastai.basics import *

class UndoDict(ItemTransform):
    split_idx = None

    def __init__(self, keys=["image", "label"], do_cat=False):
        self.keys = keys
        self.do_cat = do_cat
        
    def encodes(self, d):
        item = tuple(d[key] for key in self.keys)
        # concatenate inputs into channel, label seperately
        if self.do_cat: item = torch.cat(item[:-1], dim=0), item[-1]
        return item

    def __str__(self):
        return f"UndoDict({self.keys})"

class AddAtlas(ItemTransform):
    split_idx = None
    
    # items are train items from ABIDE
    def __init__(self, items):
        self.n = len(items)
        self.items = items
    
    def encodes(self, item):
        atlas_item = self.items[np.random.randint(self.n)]
        while atlas_item == item:
            atlas_item = self.items[np.random.randint(self.n)]
        item["atlas_image"] = atlas_item["image"]
        item["atlas_label"] = atlas_item["label"]
        return item

    
# Simplified
def get_train_valid_transforms(pixdim=(1.5,1.5,1.5), full_res=(96, 96, 96), 
                               do_flip=False, do_simple = True, do_condseg=False, items=None):

    keys       = ["image", "label", "atlas_image", "atlas_label"] if do_condseg else ["image", "label"]
    input_keys = ["image", "label"]
    image_keys = ["image", "atlas_image"] if do_condseg else ["image"]
    atlas_keys = ["atlas_image", "atlas_label"]
    
    # reorder so 3ch input (label is seperate)
    undo_dict_keys = ["image", "atlas_image", "atlas_label", "label"] if do_condseg else keys
    
    #print(full_res, "do flip", do_flip, "do simple", do_simple, "do condseg", do_condseg)
    p = 0.5

    # load images, isotropic (all keys), normalize intensity (image keys)
    standard_load = [AddAtlas(items)] if do_condseg else []
    standard_load += [
        LoadImaged(keys),
        Spacingd(keys, pixdim=pixdim, mode=("bilinear", "nearest")*(len(keys)//2)),
        NormalizeIntensityd(image_keys, nonzero=True, channel_wise=False),
    ]
    
    # augment intensity/noise/contrast (only input image)
    rand_intensity_augs = [] if do_simple else [
        RandScaleIntensityd(input_keys[0], factors=0.1, prob=p),
        RandShiftIntensityd(input_keys[0], offsets=0.1, prob=p),
        RandGaussianNoised(input_keys[0], prob=p, std=0.01), # prob = 0.15
        RandAdjustContrastd(input_keys[0], prob=p, gamma=(0.5, 2.)),
    ]
    
    # augment flip (both input image and mask)
    rand_flip_augs = [] if do_simple else [
        RandFlipd(input_keys, spatial_axis=0, prob=p),
        RandFlipd(input_keys, spatial_axis=1, prob=p),
    ]
    
    # add channel (all keys)
    add_ch = [AddChanneld(keys)]
    
    # augment affine (input image and mask)
    rand_affine = [
        RandAffined(input_keys, mode=("bilinear", "nearest"), prob=p, spatial_size = (192, 192, 192),
            rotate_range=(np.pi/8,np.pi/8,np.pi/8), 
            scale_range = (0.1, 0.1, 0.1),
            translate_range = (15, 15, 15),
            padding_mode = "border",
#                 device=torch.device('cuda:0')
        ),
    ]
    
    # in case dim < full_res (atlas was not resized in affine, or, in valid set, no one resized in affine)
    atlas_pad = [SpatialPadd(keys=atlas_keys, spatial_size=full_res, method="symmetric", mode="constant")] if do_condseg else []
    all_pad   = [SpatialPadd(keys=keys,       spatial_size=full_res, method="symmetric", mode="constant")]
    
    # crop, to tensor tuple
    standard_finish = [
        CenterSpatialCropd(keys=keys, roi_size=full_res),
        ToTensord(keys=keys),
        UndoDict(keys=undo_dict_keys, do_cat=do_condseg),
    ]
    
   
    train_transforms = Compose(standard_load + rand_intensity_augs + rand_flip_augs + add_ch + rand_affine + atlas_pad + standard_finish)
    valid_transforms = Compose(standard_load + add_ch + all_pad + standard_finish)
    
    # print(standard_finish)
    return train_transforms, valid_transforms

# printing funcs
def get_tfm_d(tfm):
    d = {}
    for k,v in tfm.__dict__.items():
        if k not in ("image_key", "label_ley", "allow_missing_keys"):
            d[k] = v
            try:
                d.update(v.__dict__)
                #print(v.__dict__)
            except:
                pass
    return d

# for OBELISK 144
def resample_144_to_96(pixdim = (1,5, 1.5, 1.5), full_res = (96, 96, 96)):
    keys       = ["image", "label"]
    input_keys = ["image", "label"]
    image_keys = ["image"]
    
    tfms = Compose([
        Spacingd(keys, pixdim=pixdim, mode=("bilinear", "nearest")),
        CenterSpatialCropd(keys=keys, roi_size=full_res)
    ])
    return tfms    

def monai_tfms2str(tfms):
    simple = "\n".join([str(tfm) for tfm in tfms.transforms])
    sep = "\n" + "*"*50 + "\n"
    details = sep.join([str(tfm)+"\n"+str(pformat(get_tfm_d(tfm), indent=4)) for tfm in tfms.transforms])
    return simple + sep + details


def fake_tfms(key, pixdim=(1.5,1.5,1.5), full_res=(96, 96, 96)):
    return Compose([
        Spacingd(key, pixdim=pixdim, mode="nearest"),
        AddChanneld(key),
        SpatialPadd(key,       spatial_size=full_res, method="symmetric", mode="constant"),
        CenterSpatialCropd(key, roi_size=full_res),
        ToTensord(key),
    ])