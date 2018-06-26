import sys, os

# Directories are the labels in this convention
def get_dir_labels( 
                    inp_dir
                  ):
    # Make sure we have a directory
    if not os.path.isdir(inp_dir):
        raise ValueError( 'No such file or directory: '+inp_dir )

    # Each directory is possibly a label, double check
    label_list = []
    possible_labels = os.listdir( inp_dir )
    for label in possible_labels:
        if os.path.isdir( inp_dir + label ):
            label_list.append( label )

    return label_list

# Return list of image files
def get_white_images_labels(
                            inp_dir = '/home/sean/Desktop/lego_images_bounding_labeled/',
                            dirty   = False
                           ):
    
    # Get out list of directory labels
    label_list = get_dir_labels( inp_dir )
    
    # Output dict of labels
    label_dict = {}
    
    # Output list of image paths
    img_list = []
    for label in label_list:
        
        # Directory containing image
        img_dir = inp_dir + label + '/'
        
        # Get the images in jpg jpeg or png
        dir_contents = os.listdir( img_dir )
        for item in dir_contents:
            if (
                ( item[-4:] == '.jpg'  ) or
                ( item[-5:] == '.jpeg' ) or
                ( item[-4:] == '.png'  )
               ):
                
                # If we aren't doing the dirty images, only need extension
                # If dirty, need to make sure just taking the rescaled images
                if ( 
                    ( not dirty ) or
                    ( 'rescale' in item )
                   ):
                    img_list.append( img_dir + item )
                    label_dict[ img_dir + item ] = label
                    
    return img_list, label_dict

# For noisy images, need to do the same
#  as white images, but only for rescale
#  not for mult
def get_dirty_images_labels():
    return get_white_images_labels(
                                    inp_dir = '/home/sean/Desktop/lego_images_bounding_box_dirty/',
                                    dirty   = True
                                  )

# For noisy images, need to do the same
#  as white images, but only for rescale
#  not for mult
def get_retrain_images_labels():
    return get_white_images_labels(
                                    inp_dir = '/home/sean/Desktop/lego_dirty_close/',
                                    dirty   = True
                                  )