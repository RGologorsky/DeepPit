import torch
import torch.nn.functional as F
import numpy as np # for np.pi

# BCDHW -> BDHW -> B x (vols)
def get_vol(mk):                return mk.sum(dim=-3).sum(dim=-2).sum(dim=-1).squeeze()

def get_contour(mk, dim=3):
    # dim kernel
    kernel   = tuple(3 for _ in range(dim))
    max_pool = [0, F.max_pool1d, F.max_pool2d, F.max_pool3d][dim]
   
    # BCDHW
    min_pool_x1     = max_pool(mk.float()*-1, kernel, 1, 1)*-1
    max_min_pool_x1 = max_pool(min_pool_x1,   kernel, 1, 1)
    contour         = torch.nn.functional.relu(max_min_pool_x1 - min_pool_x1)
    
    return contour

def get_contour_vol(mk, dim=3): 
    contour = get_contour(mk, dim=dim)
    return contour, get_vol(contour)

def get_sa(mk):
    # pad to ensure no boundary issue w. contour
    mk = F.pad(mk, (1,1,1,1,1,1), "constant", 0)

    # 3d: BCDHW
    three, three_vol = get_contour_vol(mk, dim=3)
   
    # 2d: BCDHW => B(D)HW, B(H)DW, B(W)DH (channel dim = D,H,W)
    pre2d = three.squeeze(1)
    two0 = get_contour(pre2d, dim=2)
    two1 = get_contour(torch.transpose(pre2d,1,2), dim=2)
    two2 = get_contour(torch.transpose(pre2d,1,3).transpose(2,3), dim=2)

    # combine to common BCDHW and then (or them together: two0 V two1 V two2, compatible with probabilities)
    two = torch.max(two0.unsqueeze(1),
                    torch.max(
                        two1.transpose(1,2).unsqueeze(1),
                        two2.transpose(2,3).transpose(1,3).unsqueeze(1)
                    )
    )    
    two_vol = get_vol(two)

    
    # 1d: B(D)HW => (BD)HW (batch*depth = batch dim, height = channel dim)
    B,D,H,W = two0.shape
    pre1d = two0.reshape(B*D,H,W)
    one = get_contour(pre1d, dim=1).reshape(B,D,H,W)
    one_vol = get_vol(one)
    
    return three_vol + two_vol + one_vol

# sphere = 1
def get_vol_sa_ratio(vol, surfacearea): return 36 * np.pi * (vol**2/surfacearea**3)
def get_iso_ratio(mk): return get_vol_sa_ratio(get_vol(mk), get_sa(mk))