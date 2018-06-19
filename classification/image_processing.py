from PIL import Image
from PIL import ImageFilter

import numpy as np
import pandas as pd

import os


# Checks the input stuff for images is what we expect
def _check_inp(
                    resize_image   = False , # Force an image size
                    blur_image     = False , # Blur the image
                    pool_image     = False , # Crop the image
                    resize_shape   = []    , # Forced size of image if resizing, WxH
                    kernel_size    = 5     , # Box blur kernel size, 3x3 or 5x5 option
                    pool_stride    = 2     , # Stride when pooling
                    pool_grid_size = 2     , # Grid size for pooling
                    pool_method    = 'max' , # Method for pooling: max, min, avg, med
              ):

    if not (
                isinstance( resize_image, bool ) and
                isinstance(   blur_image, bool ) and
                isinstance(   pool_image, bool )
           ):
        raise TypeError( 'resize_image, blur_image, pool_image must all be of type: bool')

    pool_method = pool_method.lower()
    if  not ( 
             pool_method == 'max' or
             pool_method == 'min' or
             pool_method == 'avg' or
             pool_method == 'med'
            ):
        raise ValueError('pool_method must be max, min, med, or avg')
        
    if not (
                isinstance( kernel_size, int ) and
                ( 
                    ( kernel_size == 3 ) or
                    ( kernel_size == 5 )
                )
           ):
        raise ValueError('kernel_size must be integer of value 3 or 5')
        
    if not (
                isinstance( pool_stride, int ) and
                          ( pool_stride > 0  )
           ):
        raise ValueError('pool_stride must be an integer > 0')
        
    if not (
                isinstance( pool_grid_size, int ) and
                          ( pool_grid_size > 0  )
           ):
        raise ValueError('pool_grid_size must be an integer > 0')
        
    rs_len = False
    if isinstance( resize_shape, list ):
        if ( 
                    ( len( resize_shape )==0 ) or 
                    ( len( resize_shape )==2 )
           ):
            rs_len = True
    if not rs_len:
        raise ValueError('resize_shape must be a list of len 2')
        

# Do pooling of input numpy array
def _pool_arr( 
                inp_arr          ,  # 2d Numpy array
                grid_size = 2    ,  # Grid size for pooling, default 2
                stride    = 2    ,  # Stride for pooling, default 2
                kind      = 'max',  # How to pool the data, default max pooling
             ):
    
    # Operation
    op_dict = {
                'max': np.max     ,
                'min': np.min     ,
                'avg': np.average ,
                'med': np.median  ,
               }
    
    # Starting pixel size
    pix_0 = inp_arr.shape[0]
    pix_1 = inp_arr.shape[1]
    
    # Find resultant image size, save original size
    new_pix_0 = pix_0//stride
    new_pix_1 = pix_1//stride
    
    # Create new output image
    new_img = np.zeros( [new_pix_0,
                         new_pix_1] )

    # Loop over img and do the pooling
    # Will perform whatever function the user provided
    for     i in range( 0, new_pix_0 ):
        for j in range( 0, new_pix_1 ):


            new_img[i,j] = op_dict[kind]( 
                                         inp_arr[ 
                                                 i*stride:i*stride+grid_size,
                                                 j*stride:j*stride+grid_size
                                                ] 
                                        )
    
    return new_img

# Run multiple steps to process and alter an image
def process_image(
                    inp_image              , # Either PIL image object or path to image
                    resize_image   = False , # Force an image size
                    blur_image     = False , # Blur the image
                    pool_image     = False , # Crop the image
                    resize_shape   = []    , # Forced size of image if resizing, WxH
                    kernel_size    = 5     , # Box blur kernel size, 3x3 or 5x5 option
                    pool_stride    = 2     , # Stride when pooling
                    pool_grid_size = 2     , # Grid size for pooling
                    pool_method    = 'max' , # Method for pooling: max, min, avg, med
                    ret_pil_img    = False , # Return a PIL image instead of array
                    conv_greyscale = True  , # Convert input image to greyscale
                 ):
    
    
    # Check input variables are of types we expect
    _check_inp( resize_image, blur_image, pool_image, resize_shape, kernel_size, pool_stride, pool_grid_size, pool_method )

    
    # Read file if needed
    if   isinstance( inp_image, str ):
        
        if not os.path.exists( inp_image ):
            raise IOError( inp_image + ' is not a valid path to an image' )
        
        raw_img = Image.open( inp_image )
        
    elif isinstance( inp_image, Image.Image ):
        raw_img = inp_image
    else:
        raise TypeError('inp_image must be path to image or Pillow Image object')

    processed_img = raw_img
    if ( conv_greyscale ):
        # Convert image to greyscale
        processed_img = raw_img.convert( 'L' )

    
    # If user is resizing image, do that here
    if ( resize_image ):
        processed_img = processed_img.resize( resize_shape, Image.ANTIALIAS )

    # Blur the image
    if ( blur_image ):

        # Do square kernel blurring
        processed_img = processed_img.filter( 
                                                ImageFilter.Kernel( 
                                                                    [ kernel_size  ,
                                                                      kernel_size ], 
                                                                  np.ones(kernel_size**2) 
                                                                 ) 
                                            )

    # Extract the data as 1d array, reshape to image dimensions
    img_val_arr = np.array( processed_img.getdata() )
    img_shape   = processed_img.size
    img_val_arr = img_val_arr.reshape( img_shape[::-1] )

    # Pool the data
    if ( pool_image ):
        
        pool_method = pool_method.lower()
        
        img_val_arr = _pool_arr( 
                                 img_val_arr                , 
                                 grid_size = pool_grid_size , 
                                 stride    = pool_stride    ,
                                 kind      = pool_method
                               )
        
    return img_val_arr

# Flips array in horizontal direction
def h_flip_array( inp_array ):
    return inp_array[:,::-1]

# Rotates in 90 deg increments
def rotate_array( inp_array, deg ):
    return np.rot90( inp_array, deg )


# Gets 2d arrays out of the pixels in the df
def arr_list_from_pix_df(
                            inp_df,
                            inp_size
                        ):
    
    pixel_cols = [ col for col in inp_df.columns.values if ('pixel_'     in col ) ]

    img_list = []
    
    for ind in inp_df.index.values:
        arr_1d = inp_df.loc[ind,pixel_cols].values
        img_list.append( np.array( arr_1d ).reshape(inp_size[0],inp_size[1]) )
        
    return img_list