# Metrics Helper fns
# From: FAIMED3D and https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
#
#  1. NP: compute dice coeff, compute coverage coeff
#  2. Torch: Dice, Dice Score, Dice Loss, DiceCE, log_cosh_dice_loss

import numpy as np
import torch

def dice(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection) /
           (iflat.sum() + tflat.sum()))

def dice_score(input, target):
    return dice(input.argmax(1), target)

def dice_loss(input, target): 
    return 1 - dice(input.softmax(1)[:, 1], target)

def ce_loss(input, target):
    return torch.nn.BCEWithLogitsLoss()(input[:, 1], target.squeeze(1))
    
def dice_ce_loss(input, target):
    return dice_loss(input, target) + ce_loss(input, target)

# from https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
def log_cosh_dice_loss(input, target):
    x = dice_loss(input, target)
    return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
    
def isoperim_loss(mk):
    return 1 - get_iso_ratio(mk)

def log_cosh_dice_isoperim_loss(input, target):
    x = log_cosh_dice_loss
    return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
    
# Metrics
# Source: https://www.programcreek.com/python/?CodeExample=compute+dice
def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  mask_gt   = mask_gt.astype(bool)
  mask_pred = mask_pred.astype(bool)

  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 

def contour3d(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = torch.nn.functional.max_pool3d(x*-1, (3, 3, 3), 1, 1)*-1
    max_min_pool_x = torch.nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour

def contour3d_loss(input, target):
    '''
    inputs shape  (batch, channel, depth, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    b, _, d, w, h = input.shape
    
    # BCDWH (pred) => BDHW (softmax prob over channel dim) => unsqueeze BCDHW
    prob   = input.softmax(1)[:,1].unsqueeze(1)
    target = target.float()

    # calculate perim lenth, summed over DHW
    cl_pred         = contour3d(prob).sum(axis=(2,3,4))
    target_skeleton = contour3d(target).sum(axis=(2,3,4))

    # mse loss
    big_pen = (cl_pred - target_skeleton) ** 2
    contour_loss = big_pen / (d * w * h)

    # average over batch, remove channel dim
    loss = contour_loss.mean(axis=0)[0]
    
    return loss

def log_cosh_dice_loss(input, target):
        x = dice_loss(input, target)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

def perim_loss(input, target, x=0.8):
    dice_loss = log_cosh_dice_loss(input, target)
    contour_loss = contour3d_loss(input, target)
    return x * dice_loss + (1.-x) * contour_loss

def compute_coverage_coefficient(mask_gt, mask_pred):
  """Computes percent of ground truth label covered in prediction.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return volume_intersect / mask_gt.sum() 