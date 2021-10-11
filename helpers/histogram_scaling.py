# source: https://github.com/kbressem/faimed3d/blob/main/faimed3d/preprocess.py#L109
# https://github.com/fastai/fastai/blob/master/fastai/medical/imaging.py#L78

import torch
from torch import Tensor, tensor

from fastai.torch_core import interp_1d

# Cell
def freqhist_bins(self:Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.reshape(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()

# Cell
def hist_scaled_pt(self:Tensor, brks=None):
    # Pytorch-only version - switch to this if/when interp_1d can be optimized
    if brks is None: brks = freqhist_bins(self)
    brks = brks.to(self.device)
    ys = torch.linspace(0., 1., len(brks)).to(self.device)
    return self.flatten().interp_1d(brks, ys).reshape(self.shape).clamp(0.,1.)