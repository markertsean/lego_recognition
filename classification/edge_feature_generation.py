import os, sys

import numpy  as np
import pandas as pd

from scipy import stats
from PIL   import Image, ImageFilter

import matplotlib.pyplot as plt

def arr_from_pil( inp_image ):
    channels = np.array( inp_image )
    r_ch = channels[:,:,0].reshape( inp_image.size[1], inp_image.size[0] )
    g_ch = channels[:,:,1].reshape( inp_image.size[1], inp_image.size[0] )
    b_ch = channels[:,:,2].reshape( inp_image.size[1], inp_image.size[0] )
    
    return np.swapaxes( np.swapaxes( np.array([r_ch,g_ch,b_ch]), 0, 2 ), 0, 1 )

# Simple resize and blurring of the image
def resize_blur(
                    inp_image              ,
                    size        = [512,512],
                    kernel_size =        5 ,
               ):
    
    # Read file if needed
    if   isinstance( inp_image, str ):
        if not os.path.exists( inp_image ):
            raise IOError( inp_image + ' is not a valid path to an image' )        
        raw_img = Image.open( inp_image )        
    elif isinstance( inp_image, Image.Image ):
        raw_img = inp_image
    elif isinstance( inp_image, np.ndarray ):
        raw_img = Image.fromarray( inp_image, 'RGB' )
    else:
        raise TypeError('inp_image must be path to image or Pillow Image object')

    processed_img = raw_img.convert('RGBA')
        
    # Resizing image
    processed_img = processed_img.resize( size, Image.ANTIALIAS )

    # Blur the image
    # Do square kernel blurring
    processed_img = processed_img.filter( 
                                            ImageFilter.Kernel( 
                                                                [ kernel_size  ,
                                                                  kernel_size ], 
                                                              np.ones(kernel_size**2) 
                                                              ) 
                                        )

    img_arr = arr_from_pil( processed_img )
    return img_arr

def get_modes( 
                inp_arr ,
                n_final=5,
             ):
    count_dict = {}
    
    for     i in range( 0, inp_arr.shape[0] ):
        for j in range( 0, inp_arr.shape[1] ):
            
            my_key = str(inp_arr[i,j,:])
            if ( my_key in count_dict.keys() ):
                count_dict[ my_key ] = count_dict[ my_key ] + 1
            else:
                count_dict[ my_key ] = 0
        
    ret_list = []
    
    # Got the count of values, now to sort
    for key, value in sorted( count_dict.iteritems(), key=lambda (k,v): (v,k)):
        ret_list.append( key )
#        print "%s: %s" % (key, value)

    n_return = len(ret_list)//2
    return ret_list[-n_final:]
    
    
# Pull out the color of the lego
#  and find indexes where the color is
#  present
def color_select( 
                    img_arr,
                    val_mod   = 7,
                    n_final   = 20,
                    tolerance = 20,
                    center_bound = 0.1,
                    display        = False,
                ):
    inp_arr = img_arr.copy() // val_mod

    center_x_1 = int( img_arr.shape[0]//2 - center_bound * img_arr.shape[0] )
    center_x_2 = int( img_arr.shape[0]//2 + center_bound * img_arr.shape[0] )
    center_y_1 = int( img_arr.shape[1]//2 - center_bound * img_arr.shape[1] )
    center_y_2 = int( img_arr.shape[1]//2 + center_bound * img_arr.shape[1] )
    

    # For each channel, get mode of 
    #  center color. Should be brick color
    center_arr = inp_arr[
        center_x_1:center_x_2,
        center_y_1:center_y_2,
        :
    ]

    # Get the modes of color in a much lower resolution
    common_colors = get_modes( center_arr, n_final )        
    mode_list = []
    for i in range( 0, len(common_colors) ):
        mode_list.append( [int(val) for val in common_colors[i].strip('[').strip(']').replace('  ',' ').split(' ')] )
    mode_list = np.array( mode_list ) * val_mod
    mode_arr = np.array( mode_list )
        
    # Get mask of the pixels in all of the possible common
    # color combinations
    color_mask = np.zeros( inp_arr.shape[:2], dtype=bool )
    for     i in range( 0, mode_arr.shape[0] ):
        color_channel_mask = np.ones( inp_arr.shape[:2], dtype=bool )
        for j in range( 0, 3 ): 
            this_channel = (
                                    ( inp_arr[:,:,j]*val_mod >= (mode_arr[i,j]-tolerance) ) & 
                                    ( inp_arr[:,:,j]*val_mod <= (mode_arr[i,j]+tolerance) ) 
                            )
            color_channel_mask = color_channel_mask & this_channel        
        color_mask = color_mask | color_channel_mask

    masked_arr = img_arr.copy()
    masked_arr[~color_mask,0] = 255
    masked_arr[~color_mask,1] = 255
    masked_arr[~color_mask,2] = 255
    
    #
    if( display ):
        foo = np.where( color_mask, 1, 0 )
        plt.imshow( foo )
        plt.show()

        plt.imshow( masked_arr )
        plt.show()
    
    return color_mask, masked_arr

# Returns slope of lego
# Note this doesnt match picture, as y axis flipped
def lego_orientation(
                        inp_mask,
                    ):
    
    from sklearn.linear_model import LinearRegression
    
    # Mask is same for all channels,
    #  so can just use the x and y vals
    x_vals = []
    y_vals = []
    
    for     i in range( 0, inp_mask.shape[0] ):
        for j in range( 0, inp_mask.shape[1] ):
            
            if ( inp_mask[i,j] ):
                x_vals.append( i )
                y_vals.append( j )
    
    try:
        reg = LinearRegression()
        reg.fit( np.array(x_vals).reshape(-1,1), np.array(y_vals).reshape(-1,1) )

        return reg.coef_[0][0], reg.intercept_[0]
    except:
        return None, None
    
def reorient_lego(
                    inp_img,
                    slope,
                    intercept,
                    display = False,
                 ):
    
    angle   = np.arctan( -slope/2 ) * 360 / np.pi
    
    inp_img = Image.fromarray( inp_img, 'RGB' )
    
    rot_img = inp_img.rotate(-angle)
    
    rot_arr = arr_from_pil( rot_img )
    
    # Fill white
    rot_arr[
            ( rot_arr[:,:,0] == 0 ) & 
            ( rot_arr[:,:,1] == 0 ) & 
            ( rot_arr[:,:,2] == 0 ) 
           ] = 255
    
    
    if( display ):
        fig, ax = plt.subplots(1)
        ax.imshow( inp_img )
        ax.plot( range(0,inp_img.size[0]), np.arange(0,inp_img.size[0])*slope+intercept)
        plt.show()
        
        plt.imshow(rot_arr)
        plt.show()
    
    return rot_arr

# Get the edge features from images
def locate_edges(
                    inp_arr,
                    square  = False,
                    display = False,
                    smooth  = False,
                ):
    
    display=True
    # Change to image
    inp_img  = Image.fromarray( inp_arr, 'RGB' )

    if ( int(smooth) != 0 ):
        inp_img = inp_img.filter( 
                                    ImageFilter.Kernel( 
                                                        [ smooth  ,
                                                          smooth ], 
                                                      np.ones(smooth**2) 
                                                      ) 
                                )

    
    # Locate edges
    edge_img =  inp_img.filter(ImageFilter.FIND_EDGES)
    
    # Convert to np array
    edge_arr = arr_from_pil( edge_img )

    # Average pixel values
    if ( square ):
        grey_edge_arr = np.sqrt( np.sum( edge_arr**2, axis=2 ) ) / 3
    else:
        grey_edge_arr =          np.sum( edge_arr , axis=2 )
        
    if ( display ):
        plt.imshow( grey_edge_arr )
        plt.show()
        
    return grey_edge_arr

# Engineer features from the edges
def get_edge_features(
                        inp_arr,
                        display = False,
                     ):
    
    # Get the sum of rows and columns
    # Ignore very edges
    inp_arr_row = np.sum( inp_arr[1:-1,1:-1], axis=1 )
    inp_arr_col = np.sum( inp_arr[1:-1,1:-1], axis=0 )

    # Normalizations
    row_norm    = np.average( inp_arr_row )
    col_norm    = np.average( inp_arr_col )

    # Do normalization
    inp_arr_row = inp_arr_row / row_norm
    inp_arr_col = inp_arr_col / col_norm
    
    # Flip row and col, if col is shorter
    #  than row. Want it the long axis
    col_first_index =                        np.argmax( inp_arr_col       > 0.5 )
    col_last_index  = inp_arr_col.shape[0] - np.argmax( inp_arr_col[::-1] > 0.5 )
    row_first_index =                        np.argmax( inp_arr_row       > 0.5 )
    row_last_index  = inp_arr_row.shape[0] - np.argmax( inp_arr_row[::-1] > 0.5 )
    
    # Get the width of each
    col_width = col_last_index - col_first_index
    row_width = row_last_index - row_first_index
        
    if ( col_width < row_width ):
        foo = inp_arr_row.copy()
        inp_arr_row = inp_arr_col.copy()
        inp_arr_col = foo.copy()
        
        foo       = col_width + 0.
        col_width = row_width + 0.
        row_width = foo       + 0.
    
    
    if ( display ):
        plt.plot( range(0,inp_arr_col.shape[0]), inp_arr_col )
        plt.title('Col')
        plt.show()

        plt.plot( range(0,inp_arr_row.shape[0]), inp_arr_row )
        plt.title('Row')
        plt.show()
        
    return row_width/float(col_width), inp_arr_row, inp_arr_col

def get_img_edge_data(
                        inp_file,
                        size           = [516,516],
                        blur           =  5  ,
                        n_color_groups =  6  ,
                        tolerance      = 20  ,
                        n_mode_iter    = 30  ,
                        center_size    =  0.1,
                        display        = False,
                        square         = False,
                        edge_smooth    = False,
                        display_orientation=False,
                     ):
    
    # Smooth the image
    pp_img_arr = resize_blur( inp_file, size, blur )

    
    # Get indexes containing lego
    color_mask, masked_arr = color_select( 
                                                pp_img_arr,
                                                val_mod        = n_color_groups  ,
                                                tolerance      = tolerance       ,
                                                center_bound   = center_size     ,
                                                n_final        = n_mode_iter     ,
                                                display        = display         ,
                                         )

    # Inclination, and axis intercept, of lego image
    slope, intercept = lego_orientation( 
                                            color_mask,
                                       )
    
    # Rotate so lego horizontal
    rot_arr = reorient_lego( 
                                masked_arr, 
                                slope, 
                                intercept, 
                                display = (display or display_orientation),
                          )

    # Get the edges
    edge_arr = locate_edges( 
                                rot_arr, 
                                square=square,
                                smooth=edge_smooth,
                                display=display
                           )

    # Get the edge features out
    r_c, row_vals, col_vals = get_edge_features( 
                                                    edge_arr,
                                                    display=display,
                                               ) 
    
    return r_c, row_vals, col_vals