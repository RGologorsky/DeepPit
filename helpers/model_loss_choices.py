# Obelisk
from helpers.items_constants import deepPit_src, obelisk_src

import sys

sys.path.append(deepPit_src)
sys.path.append(obelisk_src)

# OBELISK
from utils  import *
from models import obelisk_visceral, obeliskhybrid_visceral

# my losses dice_score, dice_loss, ce_loss, dice_ce_loss, log_cosh_dice_loss, perim_loss
from helpers.losses import *

# torch
import torch

# MONAI
from monai.losses        import DiceLoss
from monai.metrics       import DiceMetric
from monai.networks.nets import VNet, UNet, SegResNetVAE, HighResNet, UNETR


def get_model(model_type, full_res, dropout_rate=0.0):
    if model_type == "VNET":
        # https://docs.monai.io/en/latest/networks.html#vnet
        model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
        )

    elif model_type == "UNET3D":
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=dropout_rate,
        )
        
    elif model_type == "CONDSEG":
        model = UNet(
            dimensions=3,
            in_channels=3,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=dropout_rate,
        )

    elif model_type == "UNETR":
        model = UNETR(
            in_channels=1, 
            out_channels=2, 
            img_size=full_res, 
            feature_size=16, 
            hidden_size=768, 
            mlp_dim=3072, 
            num_heads=12, 
            pos_embed='perceptron', 
            norm_name='instance', 
            conv_block=False, 
            res_block=True, 
            dropout_rate=dropout_rate,
        )

    elif model_type == "SegResNetVAE":
        model = SegResNetVAE(
            input_image_size = full_res, 
            vae_estimate_std=False, 
            vae_default_std=0.3, 
            vae_nz=256, 
            spatial_dims=3, 
            init_filters=8, 
            in_channels=1, 
            out_channels=2, 
            dropout_prob=None, 
            act=('RELU', {'inplace': True}), 
            norm=('GROUP', {'num_groups': 8}), 
            use_conv_final=True, 
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), #upsample_mode=<UpsampleMode.NONTRAINABLE: 'nontrainable'>
        )

    elif model_type == "OBELISKHYBRID":
        # obelisk
        model    = obeliskhybrid_visceral(num_labels=2, full_res=full_res)
        
    else:
        print("Model not recognized.")
        
    return model


def get_loss(loss_type):
    if loss_type == "log_cosh_dice_loss":
        loss_function = log_cosh_dice_loss
        
    elif loss_type == "DICE_loss":
        loss_function = dice_loss

    elif loss_type == "BCE_loss":
        loss_function = ce_loss
        
    elif loss_type == "perim_loss":
        loss_function = log_cosh_dice_loss
        #loss_function = perim_loss

    elif loss_type == "vae_loss":
        loss_function = vae_loss
        
    else:
        print("Loss type not recognized.")
        loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=False)
        
    return loss_function