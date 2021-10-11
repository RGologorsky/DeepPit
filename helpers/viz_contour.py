# Helper functions
from helpers.losses     import contour3d, contour3d_loss
from helpers.preprocess import mask2bbox, print_bbox
from helpers.general    import sitk2np, np2sitk, print_sitk_info, lrange


# Input IO
import SimpleITK as sitk

# Numpy and Pytprch
import torch
import numpy as np

# Fastai
from fastai import *
from fastai.basics import *

# 3D
from mpl_toolkits import mplot3d
from matplotlib import colors
#%matplotlib inline

# Source: https://github.com/kbressem/faimed3d/blob/main/nbs/01_basics.ipynb

from torch import Tensor

def bbox_crop(im, bbox, axis_idx=None, margin=5):
    
    a,b,c,d,e,f = bbox
    
    # margin
    a -= margin; c -= margin; e -= margin
    b += margin; d += margin; f += margin
    
    
    if axis_idx is None:
        return im[a:b, c:d, e:f]
    
    elif axis_idx == 0:
        return im[c:d, e:f]
    
    elif axis_idx == 1:
        return im[a:b, e:f]
    
    elif axis_idx == 2:
        return im[a:b, c:d]
    
    else:
        return None
    
# export
@patch
def _strip_along(x:Tensor, dim):
    return x
#     slices = torch.unbind(x, dim)
#     slices = [s for s in slices if s.sum() != 0]
#     out = torch.stack(slices, dim)
#     return out

@patch
def strip(x:Tensor):
    return x._strip_along(-1)._strip_along(-2)._strip_along(-3)

@patch
def create_mesh(self:Tensor, cl, color, alpha):
    "creates a 3D mesh for a single class in the 3D Tensor"
    if self.ndim != 3: raise NotImplementedError('Currently only rank 3 tensors are supported for rendering')
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    im = (self == cl).float()
    verts, faces, normals, values = marching_cubes(im.permute(1, 2, 0).numpy())
    mesh = Poly3DCollection(verts[faces])
    mesh.set_facecolor(color)
    mesh.set_alpha(alpha)
    return mesh

@patch
def show_center_point(self:Tensor):
    "displays a cross in the center of the mask"
    z, x, y = self.get_nonzero_bool()
    center_z = self.get_center_id(z)
    center_x = self.get_center_id(x)
    center_y = self.get_center_id(y)
    c_val = im.max()+2
    self[center_z, center_x-50:center_x+50, center_y-5:center_y+5] = c_val
    self[center_z, center_x-5:center_x+5, center_y-50:center_y+50] = c_val
    self.show()
@patch
def create_mesh_for_classes(self:Tensor, colors, alpha):
    "applies `create_mesh` to all classes of the mask"
    classes = self.unique()[1:]
    if colors is None: colors = 'bgrcmyw'
    if len(classes) > len(colors):
        colors = random.choices(colors, k=len(classes))
    else:
        colors = colors[0:len(classes)]
    if alpha is None: alpha = (0.5, )*len(classes)
    if type(alpha) is not tuple: raise TypeError('alpha need to be a tuple.')
    if len(alpha) == 1: alpha = alpha*len(classes)
    if len(alpha) != len(classes):
        raise ValueError('Number of classes and number of alpha does not match')
    cca = list(zip(classes, colors, alpha))
    meshes = [self.create_mesh(cl=cl, color=color, alpha=alpha) for cl, color, alpha in cca]
    return meshes
@patch
def render_3d(self:Tensor, colors=None, alpha = None, symmetric=False):
    "renders the mask as a 3D object and displays it"

    im = self.strip()
    meshes = im.create_mesh_for_classes(colors = colors, alpha = alpha)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for m in meshes: ax.add_collection3d(m)
    ax.set_xlim(0, im.size(1))
    ax.set_ylim(0, im.size(2))
    ax.set_zlim(0, im.size(0))
    ax.set_facecolor('k')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
@patch
def calc_volume(self:Tensor):
    """
    Calculates the volume for a single class in the mask.
    Calculation relies on correct spacing information in the header.
    Results are given in mm**3
    """
    x,y,z = 1,1,1
    voxel_size = x*y*z
    self.volume = {'background': self._calc_vol_per_class(0, voxel_size)}
    self.volume['total_mask_volume'] = self.size(0)*self.size(1)*self.size(2)*voxel_size - self.volume['background']
    for c in self.unique()[1:]:
        name = 'class '+str(int(c))
        self.volume[name] = self._calc_vol_per_class(c, voxel_size)
    #print(self.volume)
    return self.volume["class 1"]

@patch
def _calc_vol_per_class(self:Tensor, class_idx, voxel_size):
    "calculates volume of the void, whole mask and for each class in the mask"
    return float((self == class_idx).sum() * voxel_size)

def plot3d(y, contour, min_pool_x1, max_min_pool_x1):
    fig = plt.figure()

    ax1 = fig.add_subplot(141, projection='3d')
    ax2 = fig.add_subplot(142, projection='3d')
    ax3 = fig.add_subplot(143, projection='3d')
    ax4 = fig.add_subplot(144, projection='3d')

    axes = [ax1, ax2, ax3, ax4]
    ims  = [y, contour, min_pool_x1, max_min_pool_x1]
    titles = ["Target", "Contour", "Erosion", "Dilation"]

    colors=None
    alpha = None

    for ax, im, title in zip(axes, ims, titles):
        im = im.strip()
        meshes = im.create_mesh_for_classes(colors = colors, alpha = alpha)
        for m in meshes: ax.add_collection3d(m)
        ax.set_xlim(0, im.size(1))
        ax.set_ylim(0, im.size(2))
        ax.set_zlim(0, im.size(0))
        ax.set_facecolor('k')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def process3d(x,y, do_plot=False):
    bbox = mask2bbox(np.asarray(y))

    # Crop
    x_crop = bbox_crop(x,bbox)
    y_crop = bbox_crop(y, bbox)

    # calculate 3d
    x1 = y_crop.float().unsqueeze(0).unsqueeze(0)
    min_pool_x1     = torch.nn.functional.max_pool3d(x1*-1, (3, 3, 3), 1, 1)*-1
    max_min_pool_x1 = torch.nn.functional.max_pool3d(min_pool_x1, (3, 3, 3), 1, 1)
    contour         = torch.nn.functional.relu(max_min_pool_x1 - min_pool_x1)

    contour         = contour.squeeze()
    min_pool_x1     = min_pool_x1.squeeze()
    max_min_pool_x1 = max_min_pool_x1.squeeze()

    #print(type(min_pool_x1))
    
    # print side lens
    y_vol = y.calc_volume()
    min_vol = min_pool_x1.calc_volume()
    max_vol = max_min_pool_x1.calc_volume()
    contour_vol = contour.calc_volume()

    # side lens
    #print(bbox)
    side1 = bbox[1]-bbox[0]
    side2 = bbox[3]-bbox[2]
    side3 = bbox[5]-bbox[4]

    # correction for idx 2
    # side1 -= 1

    volume = side1 * side2 * side3
    surface_area = 2 * (side1 * side2) + 2 * (side1*side3) + 2 * (side2 * side3)

    diff_vol = volume-y_vol
    diff_sa  = surface_area - contour_vol

    # Plot
    if do_plot: 
        plot3d(y, contour, min_pool_x1, max_min_pool_x1)    
    return (side1, side2, side3), diff_vol, diff_sa
        
    # print
#     print(f"Bbox: {bbox}")
#     print(side1, side2, side3)
#     print(f"Vol: GT", volume, "Prism", y_vol, "Diff", volume-y_vol)
#     print(f"SA: GT", surface_area, "Prism ", contour_vol, "Diff: ", surface_area - contour_vol)
    
def plot2d(x,y):
    bin_cmap2  = colors.ListedColormap(['white', 'red'])

    bbox = mask2bbox(np.asarray(y))
    #print("Bbox", bbox)

    slice_ranges = [(bbox[2*i], bbox[2*i+1]) for i in range(3)]

    # take slice
    for axis_idx in range(3):
        slice_start, slice_end = slice_ranges[axis_idx]
        #slice_mid = slice_start + (slice_end-slice_start)//2
        #slice_start, slice_end = slice_mid-2, slice_mid+2

        for slice_idx in range(slice_start, slice_end):
            #slice_idx = slice_start + (slice_end - slice_start)//2

            x_slice = np.take(np.asarray(x), slice_idx, axis=axis_idx)
            y_slice = np.take(np.asarray(y), slice_idx, axis=axis_idx)

            # calculate  2d
            x1 = torch.tensor(y_slice).float().unsqueeze(0).unsqueeze(0)
            min_pool_x1     = torch.nn.functional.max_pool2d(x1*-1, (3, 3), 1, 1)*-1
            max_min_pool_x1 = torch.nn.functional.max_pool2d(min_pool_x1, (3, 3), 1, 1)
            contour         = torch.nn.functional.relu(max_min_pool_x1 - min_pool_x1)

            min_pool_x1 = np.asarray(min_pool_x1.squeeze())
            max_min_pool_x1 = np.asarray(max_min_pool_x1.squeeze())
            y_slice_contour = np.asarray(contour.squeeze().long())

            _, axes = plt.subplots(nrows=1, ncols=5)
            axes[0].imshow(np.rot90(x_slice))
            axes[0].imshow(np.rot90(y_slice), alpha=0.5, cmap=bin_cmap2)
            axes[1].imshow(np.rot90(bbox_crop(y_slice, bbox, axis_idx)), cmap=bin_cmap2)
            axes[2].imshow(np.rot90(bbox_crop(y_slice_contour, bbox, axis_idx)), cmap=bin_cmap2)
            axes[3].imshow(np.rot90(bbox_crop(min_pool_x1, bbox, axis_idx)), cmap=bin_cmap2)
            axes[4].imshow(np.rot90(bbox_crop(max_min_pool_x1, bbox, axis_idx)), cmap=bin_cmap2)

            axes[0].set_title(f"Slice {slice_idx} (Axis {axis_idx})")
            axes[1].set_title(f"Target")
            axes[2].set_title(f"Contour")
            axes[3].set_title(f"Erosion")
            axes[4].set_title(f"Dilation")
            
            plt.show()