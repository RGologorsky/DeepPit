# thanks to FAIMED3D 02_preprocessing
# 1. get_percentile (gets kth percentile of values in tensor)
# 2. get_landmarks (gets decile landmarks)
# 3. find_standard_scale (averages the landmark values over input tensors)
# 4. piecewise_hist (input vals => input landmarks => standard scale landmarks => standard scale vals)

# fastAI decorators
from fastai.basics import *


# PyTorch
import torch
from torch import Tensor

def get_percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (float).

    This function is twice as fast as torch.quantile and has no size limitations
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k)[0].item()

    return result

def get_landmarks(t: torch.Tensor, percentiles: torch.Tensor)->torch.Tensor:
    """
    Returns the input's landmarks.

    :param t (torch.Tensor): Input tensor.
    :param percentiles (torch.Tensor): Peraentiles to calculate landmarks for.
    :return: Resulting landmarks (torch.tensor).
    """
    return tensor([get_percentile(t, perc.item()) for perc in percentiles])

def find_standard_scale(inputs, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images
    Args:
        inputs (list or L): set of TensorDicom3D objects which are to be normalized
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)
    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    percs = torch.cat([torch.tensor([i_min]),
                       torch.arange(l_percentile, u_percentile+1, step),
                       torch.tensor([i_max])], dim=0)
    standard_scale = torch.zeros(len(percs))

    for input_image in inputs:
        mask_data = input_image > input_image.mean()
        masked = input_image[mask_data]
        landmarks = get_landmarks(masked, percs)
        min_p = get_percentile(masked, i_min)
        max_p = get_percentile(masked, i_max)
        new_landmarks = landmarks.interp_1d(torch.FloatTensor([i_s_min, i_s_max]),
                                            torch.FloatTensor([min_p, max_p]))
        standard_scale += new_landmarks
    standard_scale = standard_scale / len(inputs)
    return standard_scale, percs

@patch
def piecewise_hist(image:torch.Tensor, landmark_percs, standard_scale)->torch.Tensor:
    """
    Do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

    Args:
        input_image (TensorDicom3D): image on which to find landmarks
        landmark_percs (torch.tensor): corresponding landmark points of standard scale
        standard_scale (torch.tensor): landmarks on the standard scale
    Returns:
        normalized (TensorDicom3D): normalized image
    """
    mask_data = image > image.mean()
    masked = image[mask_data]
    landmarks = get_landmarks(masked, landmark_percs)
    if landmarks.device != image.device: landmarks = landmarks.to(image.device)
    if standard_scale.device != image.device: standard_scale = standard_scale.to(image.device)
    return image.flatten().interp_1d(landmarks, standard_scale).reshape(image.shape)