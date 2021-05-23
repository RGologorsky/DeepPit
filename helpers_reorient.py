# This file contains functions to re-orient sitk obj to LPS coord

# imports
import itk
import SimpleITK as sitk
import numpy as np

from itk_orient_defs import \
ITK_COORDINATE_ORIENTATION_RAS, \
ITK_COORDINATE_ORIENTATION_LPS, \
ITK_COORDINATE_ORIENTATION_LSP, \
ITK_COORDINATE_ORIENTATION_RSP, \
ITK_COORDINATE_ORIENTATION_RSA, \
ITK_COORDINATE_ORIENTATION_RPI, \
ITK_COORDINATE_ORIENTATION_LPI, \
ITK_COORDINATE_ORIENTATION_RAI, \
ITK_COORDINATE_ORIENTATION_LAI, \
ITK_COORDINATE_ORIENTATION_RPS \

# https://github.com/sbonaretti/pyKNEEr/blob/master/pykneer/pykneer/sitk_functions.py

standard_dir = (1,0,0,0,1,0,0,0,1)

def reorient(img, new_orientation=ITK_COORDINATE_ORIENTATION_LPI, new_dir = standard_dir):

    # this function is implemented using ITK since SimpleITK has not implemented the filter "OrientImageFilter" yet
    # the ITK orientation system is from this blog post: https://itk.org/pipermail/insight-users/2017-May/054606.html
    # comparison ITK - simpleITK filters: https://itk.org/SimpleITKDoxygen/html/Filter_Coverage.html
    # see also: https://github.com/fedorov/lidc-idri-conversion/blob/master/seg/seg_converter.py

    # change image name
    img_sitk = img

    # get characteristics of simpleITK image
    size_in_sitk      = img_sitk.GetSize()
    spacing_in_sitk   = img_sitk.GetSpacing()
    origin_in_sitk    = img_sitk.GetOrigin()
    direction_in_sitk = img_sitk.GetDirection()

    # allocate ITK image (type and size)
    Dimension   = 3
    PixelType   = itk.F
    ImageTypeIn = itk.Image[PixelType, Dimension]
    img_itk = ImageTypeIn.New()
    sizeIn_itk = itk.Size[Dimension]()
    for i in range (0,Dimension):
        sizeIn_itk[i] = size_in_sitk[i]
    region = itk.ImageRegion[Dimension]()
    region.SetSize(sizeIn_itk)
    img_itk.SetRegions(region)
    img_itk.Allocate()

    # pass image from simpleITK to numpy
    img_py  = sitk.GetArrayFromImage(img_sitk)

    # pass image from numpy to ITK
    img_itk = itk.GetImageViewFromArray(img_py)

    # pass characteristics from simpleITK image to ITK image (except size, assigned in allocation)
    spacing_in_itk = itk.Vector[itk.F, Dimension]()
    for i in range (0,Dimension):
        spacing_in_itk[i] = spacing_in_sitk[i]
    img_itk.SetSpacing(spacing_in_itk)

    origin_in_itk  = itk.Point[itk.F, Dimension]()
    for i in range (0,Dimension):
        origin_in_itk[i]  = origin_in_sitk[i]
    img_itk.SetOrigin(origin_in_itk)

    # old way of assigning direction (until ITK 4.13)
#     direction_in_itk = itk.Matrix[itk.F,3,3]()
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(0,0,direction_in_sitk[0]) # r,c,value
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(0,1,direction_in_sitk[1])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(0,2,direction_in_sitk[2])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(1,0,direction_in_sitk[3]) # r,c,value
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(1,1,direction_in_sitk[4])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(1,2,direction_in_sitk[5])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(2,0,direction_in_sitk[6]) # r,c,value
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(2,1,direction_in_sitk[7])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(2,2,direction_in_sitk[8])

    direction_in_itk = np.eye(3)
    direction_in_itk[0][0] = direction_in_sitk[0]
    direction_in_itk[0][1] = direction_in_sitk[1]
    direction_in_itk[0][2] = direction_in_sitk[2]
    direction_in_itk[1][0] = direction_in_sitk[3]
    direction_in_itk[1][1] = direction_in_sitk[4]
    direction_in_itk[1][2] = direction_in_sitk[5]
    direction_in_itk[2][0] = direction_in_sitk[6]
    direction_in_itk[2][1] = direction_in_sitk[7]
    direction_in_itk[2][2] = direction_in_sitk[8]
    img_itk.SetDirection(itk.matrix_from_array(direction_in_itk))

    # make sure image is float for the orientation filter (GetImageViewFromArray sets it to unsigned char)
    ImageTypeIn_afterPy = type(img_itk)
    ImageTypeOut        = itk.Image[itk.F, 3]
    CastFilterType      = itk.CastImageFilter[ImageTypeIn_afterPy, ImageTypeOut]
    castFilter          = CastFilterType.New()
    castFilter.SetInput(img_itk)
    castFilter.Update()
    img_itk             = castFilter.GetOutput()

    # change orientation to RAI
    OrientType = itk.OrientImageFilter[ImageTypeOut,ImageTypeOut]
    filter     = OrientType.New()
    filter.UseImageDirectionOn()
    filter.SetDesiredCoordinateOrientation(new_orientation)
    filter.SetInput(img_itk)
    filter.Update()
    img_itk    = filter.GetOutput()

    # get characteristics of ITK image
    spacing_out_itk   = img_itk.GetSpacing()
    origin_out_itk    = img_itk.GetOrigin()

    # pass image from itk to numpy
    img_py = itk.GetArrayViewFromImage(img_itk)

    # pass image from numpy to simpleitk
    img = sitk.GetImageFromArray(img_py)

    # pass characteristics from ITK image to simpleITK image (except size, implicitely passed)
    spacing = []
    for i in range (0, Dimension):
        spacing.append(spacing_out_itk[i])
    img.SetSpacing(spacing)

    origin = []
    for i in range (0, Dimension):
        origin.append(origin_out_itk[i])
        
    img.SetOrigin(origin)
    img.SetDirection(new_dir)

    return img
