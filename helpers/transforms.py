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

class ConcatInputs(ItemTransform):
    split_idx = None

    def encodes(self, tup):
        print(len(tup))
        return torch.cat(tup[:-1], dim=0), tup[-1]

    def __str__(self):
        return f"ConcatInputs()"
    
def get_train_valid_transforms_simple(pixdim=(1.0,1.0,1.0), full_res=(144, 144, 144)):
    keys = ["image", "label"]
    p = 0.5

    train_transforms = Compose(
        [
            LoadImaged(keys),
            Spacingd(keys, pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys[0], nonzero=True, channel_wise=False),
 
            AddChanneld(keys),
            RandAffined(keys, mode=("bilinear", "nearest"), prob=p, spatial_size = (192, 192, 192),
                rotate_range=(np.pi/8,np.pi/8,np.pi/8), 
                scale_range = (0.1, 0.1, 0.1),
                translate_range = (15, 15, 15),
                padding_mode = "border",
#                 device=torch.device('cuda:0')
            ),
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(),
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=keys),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys[0], nonzero=True, channel_wise=False),
            AddChanneld(keys=keys),
            SpatialPadd(keys=keys, spatial_size=full_res, method="symmetric", mode="constant"), # in case dim < full_res
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(),
        ]
    )
    
    return train_transforms, val_transforms
    
def get_train_valid_transforms(pixdim=(1.5,1.5,1.5), full_res=(96, 96, 96), do_flip=True):
    keys = ["image", "label"]
    p = 0.5

    train_transforms = \
        [
            LoadImaged(keys),
            Spacingd(keys, pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys[0], nonzero=True, channel_wise=False),
                        
            RandScaleIntensityd(keys[0], factors=0.1, prob=p),
            RandShiftIntensityd(keys[0], offsets=0.1, prob=p),
            RandGaussianNoised(keys[0], prob=p, std=0.01), # prob = 0.15
            RandAdjustContrastd(keys[0], prob=p, gamma=(0.5, 2.)),
            
            RandFlipd(keys, spatial_axis=0, prob=p),
            RandFlipd(keys, spatial_axis=1, prob=p),
#             RandFlipd(keys, spatial_axis=2, prob=p),
            
            AddChanneld(keys),
            RandAffined(keys, mode=("bilinear", "nearest"), prob=p, spatial_size = (192, 192, 192),
                rotate_range=(np.pi/8,np.pi/8,np.pi/8), 
                scale_range = (0.1, 0.1, 0.1),
                translate_range = (15, 15, 15),
                padding_mode = "border",
#                 device=torch.device('cuda:0')
            ),
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(),
        ]
    
    if not do_flip:
        train_transforms = train_transforms[:7] + train_transforms[9:]

    train_transforms =  Compose(train_transforms)
    
    val_transforms = Compose(
        [
            LoadImaged(keys=keys),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys[0], nonzero=True, channel_wise=False),
            AddChanneld(keys=keys),
            SpatialPadd(keys=keys, spatial_size=full_res, method="symmetric", mode="constant"), # in case dim < full_res
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(),
        ]
    )
    
    return train_transforms, val_transforms

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

def monai_tfms2str(tfms):
    simple = "\n".join([str(tfm) for tfm in tfms.transforms])
    sep = "\n" + "*"*50 + "\n"
    details = sep.join([str(tfm)+"\n"+str(pformat(get_tfm_d(tfm), indent=4)) for tfm in tfms.transforms])
    return simple + sep + details

def get_train_valid_transforms_condseg(items, pixdim=(1.5,1.5,1.5), full_res=(96, 96, 96), do_flip=True):
    keys = ["image", "label", "atlas_image", "atlas_label"]
    input_keys = ["image", "label"]
    image_keys = ["image", "atlas_image"]
    atlas_keys = ["atlas_image", "atlas_label"]
    
    p = 0.5

    # load & isotropic: all images and masks
    # intensity: only images, not masks
    # flip: only input, not atlas
    # channel: all images and masks
    # affine: only input
    # center crop: all images and masks
    # output: "image, atlas im/mk -- mask"
    
    train_transforms = \
        [
            AddAtlas(items),
            LoadImaged(keys),
            Spacingd(keys, pixdim=pixdim, mode=("bilinear", "nearest", "bilinear", "nearest")),
            NormalizeIntensityd(image_keys, nonzero=True, channel_wise=False),
                        
            RandScaleIntensityd(image_keys, factors=0.1, prob=p),
            RandShiftIntensityd(image_keys, offsets=0.1, prob=p),
            RandGaussianNoised(image_keys, prob=p, std=0.01), # prob = 0.15
            RandAdjustContrastd(image_keys, prob=p, gamma=(0.5, 2.)),
            
            RandFlipd(input_keys, spatial_axis=0, prob=p),
            RandFlipd(input_keys, spatial_axis=1, prob=p),
#             RandFlipd(keys, spatial_axis=2, prob=p),
            
            AddChanneld(keys),
            RandAffined(input_keys, mode=("bilinear", "nearest"), prob=p, spatial_size = (192, 192, 192),
                rotate_range=(np.pi/8,np.pi/8,np.pi/8), 
                scale_range = (0.1, 0.1, 0.1),
                translate_range = (15, 15, 15),
                padding_mode = "border",
#                 device=torch.device('cuda:0')
            ),
            SpatialPadd(keys=atlas_keys, spatial_size=full_res, method="symmetric", mode="constant"), # in case dim < full_res
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
        
            # reorder keys so 3ch input followed by target
            UndoDict(keys=["image", "atlas_image", "atlas_label", "label"], do_cat=True),
            # ConcatInputs()
        ]
    
    if not do_flip:
        train_transforms = train_transforms[:8] + train_transforms[10:]

    train_transforms =  Compose(train_transforms)
    
    val_transforms = Compose(
        [
            AddAtlas(items),
            LoadImaged(keys=keys),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest", "bilinear", "nearest")),
            NormalizeIntensityd(image_keys, nonzero=True, channel_wise=False),
            AddChanneld(keys=keys),
            SpatialPadd(keys=keys, spatial_size=full_res, method="symmetric", mode="constant"), # in case dim < full_res
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(keys=["image", "atlas_image", "atlas_label", "label"], do_cat=True),
            # ConcatInputs()
        ]
    )
    
    return train_transforms, val_transforms

def get_train_valid_transforms_condseg_simple(items, pixdim=(1.5,1.5,1.5), full_res=(96, 96, 96)):
    keys = ["image", "label", "atlas_image", "atlas_label"]
    input_keys = ["image", "label"]
    image_keys = ["image", "atlas_image"]
    atlas_keys = ["atlas_image", "atlas_label"]
    
    p = 0.5

    train_transforms = Compose(
        [
            AddAtlas(items),
            LoadImaged(keys),
            Spacingd(keys, pixdim=pixdim, mode=("bilinear", "nearest", "bilinear", "nearest")),
            NormalizeIntensityd(image_keys, nonzero=True, channel_wise=False),
 
            AddChanneld(keys),
            RandAffined(input_keys, mode=("bilinear", "nearest"), prob=p, spatial_size = (192, 192, 192),
                rotate_range=(np.pi/8,np.pi/8,np.pi/8), 
                scale_range = (0.1, 0.1, 0.1),
                translate_range = (15, 15, 15),
                padding_mode = "border",
#                 device=torch.device('cuda:0')
            ),
            SpatialPadd(keys=atlas_keys, spatial_size=full_res, method="symmetric", mode="constant"), # in case dim < full_res
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(keys=["image", "atlas_image", "atlas_label", "label"], do_cat=True),
            # ConcatInputs()
        ]
    )
    
    val_transforms = Compose(
        [
            AddAtlas(items),
            LoadImaged(keys=keys),
            Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest", "bilinear", "nearest")),
            NormalizeIntensityd(image_keys, nonzero=True, channel_wise=False),
            AddChanneld(keys=keys),
            SpatialPadd(keys=keys, spatial_size=full_res, method="symmetric", mode="constant"), # in case dim < full_res
            CenterSpatialCropd(keys=keys, roi_size=full_res),
            ToTensord(keys=keys),
            UndoDict(keys=["image", "atlas_image", "atlas_label", "label"], do_cat=True),
            # ConcatInputs()
        ]
    )
    
    return train_transforms, val_transforms


# Transforms
if model_type != "CONDSEG":
    if do_simple:
        train_tfms, val_tfms = get_train_valid_transforms_simple(pixdim=pixdim, full_res=full_res)
    else:
        train_tfms, val_tfms = get_train_valid_transforms(pixdim=pixdim, full_res=full_res, do_flip=do_flip)
        
else:
    train_itemsd = getd(train_items)
    if do_simple:
        train_tfms, val_tfms = get_train_valid_transforms_condseg_simple(train_itemsd, pixdim=pixdim, full_res=full_res)
    else:
        train_tfms, val_tfms = get_train_valid_transforms_condseg(train_itemsd, pixdim=pixdim, full_res=full_res, do_flip=do_flip)
