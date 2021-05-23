# This file contains defitions for ITK orientations, used to reorient to LPS standard orientation

ITK_COORDINATE_UNKNOWN = 0
ITK_COORDINATE_Right = 2
ITK_COORDINATE_Left = 3
ITK_COORDINATE_Posterior = 4
ITK_COORDINATE_Anterior = 5
ITK_COORDINATE_Inferior = 8
ITK_COORDINATE_Superior = 9 

ITK_COORDINATE_PrimaryMinor = 0
ITK_COORDINATE_SecondaryMinor = 8
ITK_COORDINATE_TertiaryMinor = 16

ITK_COORDINATE_ORIENTATION_RAS = ( ITK_COORDINATE_Right \
                                      << ITK_COORDINATE_PrimaryMinor ) \
                                    + ( ITK_COORDINATE_Anterior << ITK_COORDINATE_SecondaryMinor ) \
                                    + ( ITK_COORDINATE_Superior << ITK_COORDINATE_TertiaryMinor )

ITK_COORDINATE_ORIENTATION_LPS = ( ITK_COORDINATE_Left \
                                      << ITK_COORDINATE_PrimaryMinor ) \
                                    + ( ITK_COORDINATE_Posterior << ITK_COORDINATE_SecondaryMinor ) \
                                    + ( ITK_COORDINATE_Superior << ITK_COORDINATE_TertiaryMinor )

ITK_COORDINATE_ORIENTATION_LSP = ( ITK_COORDINATE_Left \
                                      << ITK_COORDINATE_PrimaryMinor ) \
                                    + ( ITK_COORDINATE_Superior << ITK_COORDINATE_SecondaryMinor ) \
                                    + ( ITK_COORDINATE_Posterior << ITK_COORDINATE_TertiaryMinor )


ITK_COORDINATE_ORIENTATION_RSP = ( ITK_COORDINATE_Right \
                                              << ITK_COORDINATE_PrimaryMinor ) \
                                              + ( ITK_COORDINATE_Superior << ITK_COORDINATE_SecondaryMinor ) \
                                           + ( ITK_COORDINATE_Posterior << ITK_COORDINATE_TertiaryMinor )

ITK_COORDINATE_ORIENTATION_RSA = ( ITK_COORDINATE_Right \
                                               << ITK_COORDINATE_PrimaryMinor ) \
                                        + ( ITK_COORDINATE_Superior << ITK_COORDINATE_SecondaryMinor ) \
                                           + ( ITK_COORDINATE_Anterior << ITK_COORDINATE_TertiaryMinor )

ITK_COORDINATE_ORIENTATION_RPI = (ITK_COORDINATE_Right << ITK_COORDINATE_PrimaryMinor) + \
                                (ITK_COORDINATE_Posterior << ITK_COORDINATE_SecondaryMinor) + \
                                (ITK_COORDINATE_Inferior << ITK_COORDINATE_TertiaryMinor)

ITK_COORDINATE_ORIENTATION_LPI = (ITK_COORDINATE_Left << ITK_COORDINATE_PrimaryMinor) + \
                                (ITK_COORDINATE_Posterior << ITK_COORDINATE_SecondaryMinor) + \
                                (ITK_COORDINATE_Inferior << ITK_COORDINATE_TertiaryMinor)

ITK_COORDINATE_ORIENTATION_RAI = (ITK_COORDINATE_Right << ITK_COORDINATE_PrimaryMinor) + \
                                (ITK_COORDINATE_Anterior << ITK_COORDINATE_SecondaryMinor) + \
                                (ITK_COORDINATE_Inferior << ITK_COORDINATE_TertiaryMinor)

ITK_COORDINATE_ORIENTATION_LAI = (ITK_COORDINATE_Left << ITK_COORDINATE_PrimaryMinor) + \
                                (ITK_COORDINATE_Anterior << ITK_COORDINATE_SecondaryMinor) + \
                                (ITK_COORDINATE_Inferior << ITK_COORDINATE_TertiaryMinor)

ITK_COORDINATE_ORIENTATION_RPS = (ITK_COORDINATE_Right << ITK_COORDINATE_PrimaryMinor) + \
                                (ITK_COORDINATE_Posterior << ITK_COORDINATE_SecondaryMinor) + \
                                (ITK_COORDINATE_Superior << ITK_COORDINATE_TertiaryMinor)

def get_itk_orientation(o):
    return eval(o)