import os, sys

import numpy   as np
import pandas  as pd
import cPickle as pkl

from scipy import stats
from PIL   import Image, ImageFilter

from sklearn.decomposition   import PCA
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle

import generate_image_lists as giList
import generate_image_labels as giLabels
import edge_feature_generation as efg


sys.path.append("/home/sean/Insight/legos/tensorbox")
import pred_location as pl

_DIR = '/home/sean/Insight/legos/classification/'

with open(_DIR+'data/height_logistic_clf.pkl','r') as f:
    _LOGISTIC_HEIGHT_CLF = pkl.load( f )
with open(_DIR+'data/short_logistic_clf.pkl','r') as f:
    _LOGISTIC_SHORT_CLF  = pkl.load( f )
with open(_DIR+'data/long_logistic_clf.pkl','r') as f:
    _LOGISTIC_LONG_CLF   = pkl.load( f )
with open(_DIR+'data/row_col_pca.pkl'      ,'r') as f:
    _RC_PCA              = pkl.load( f )
    
_SHORT_NUMS = [1,2,4,6,8]
_LONG_NUMS  = [1,2,3,4,6,8,10,12]



# Get the features from an image
def generate_features( img_path ):
    
    # Returns relative size of axes, and normalized sum of the rows and column
    rc_ratio, row_avg, col_avg    = efg.get_img_edge_data( img_path, blur=3 )

    # Combine the later
    row_col_arr = np.concatenate( ( row_avg, col_avg ) )
    
    # Run pca to collapse to 1/20 the original size, 85% variance
    pca_vals = _RC_PCA.transform( row_col_arr )
    
    return np.concat( ( np.array(rc_ratio), pca_vals ) )
    
    
# Will generate predictions for provided classes
# Can return raw probabilities of being the class,
#  or return the expected label
def _get_predict( 
                    inp_arr,
                    class_list,
                    clf_dict,
                    return_prob,
                ):
    
    # Get an idea of how many things we are passing
    inp_shape = len( inp_arr.shape )

    # If only one element, have to adjust format
    if ( inp_shape == 1 ):
        pred_arr_format = inp_arr.reshape(1,-1)
    else:
        pred_arr_format = inp_arr
        
    # Get the probability of a given class
    prob_dict = {}
    for classif in class_list:
        prob_dict[classif] = clf_dict[classif].predict_proba( pred_arr_format )[:,1]

    # If we are just returning the probabilities,
    #  can stop here and return a dict
    if ( return_prob ):
        return prob_dict
    
    
    # Otherwise, go through, find best prediction,
    #  and return that
    
    
    out_list = []
    
    # Compare each prediction, and 
    #  locate largest values
    # Populate the out array with these classes
    
    # Loop over each element
# LATER MODIFY TO CONSIDER THRESHOLD
    for i in range( 0, inp_arr.shape[0] ):
        
        # Loop over classes, finding the best
        best_str = class_list[0]
        for classif in class_list[1:]:
            if ( prob_dict[best_str][i] < prob_dict[classif][i] ):
                best_str = classif            
        out_list.append( best_str )
        
    return out_list
    
# Get predicted height category
def get_height_predict( 
                        inp_arr,
                        return_prob=False,
                      ):
    
    # Possible classificatios
    class_list = ['height_brick','height_plate','height_other']
    clf_dict   = _LOGISTIC_HEIGHT_CLF
    
    return _get_predict( inp_arr, class_list, clf_dict, return_prob )

# Get predicted height category
def get_short_predict( 
                        inp_arr,
                        return_prob=False,
                      ):
    
    # Possible classificatios
    class_list = ['short_'+str(col) for col in _SHORT_NUMS ]
    clf_dict   = _LOGISTIC_SHORT_CLF
    
    return _get_predict( inp_arr, class_list, clf_dict, return_prob )

# Get predicted height category
def get_long_predict( 
                        inp_arr,
                        return_prob=False,
                      ):
    
    # Possible classificatios
    class_list = ['long_'+str(col) for col in _LONG_NUMS ]
    clf_dict   = _LOGISTIC_LONG_CLF
    
    return _get_predict( inp_arr, class_list, clf_dict, return_prob )


def generate_feature_array(
                            rc_ratio,
                            row_arr,
                            col_arr,
                          ):
    row_col_arr = np.concatenate( ( row_arr, col_arr ) , axis=0 )
    rc_pca_vals = _RC_PCA.transform( row_col_arr.reshape(1,-1) )
    
    print rc_pca_vals.shape
    
    return np.concatenate( ( np.array([rc_ratio]).reshape(1,-1), rc_pca_vals ) , axis=1 )



def get_img_bbox(
                    file_name    ,
                    inp_img      ,
                    box_scale=1.0,
                ):
    
    rect_list = pl.pred_lego_locations( file_name )
    
    inp_size_x = inp_img.size[0]
    inp_size_y = inp_img.size[1]
    
    fixed_rect_dict_list = []
    
    # May return multiple
    for rect in rect_list:
                
        x_mod = inp_size_x / 640.
        y_mod = inp_size_y / 480.

        # Need to expand box since fit to 640x480 image
        # Also expand borders of box 
        x_1 =  rect['x1'] * x_mod
        x_2 =  rect['x2'] * x_mod
        y_1 =  rect['y1'] * y_mod
        y_2 =  rect['y2'] * y_mod

        width = x_2 - x_1
        height= y_2 - y_1

        # Expand to the box to really make sure we have the lego
        # But make sure we don't go out of the box
        x_1 = max( x_1 + width / 2 * ( 1 - box_scale ),             0 )
        x_2 = min( x_2 - width / 2 * ( 1 - box_scale ), inp_img.size[0]-1 )
        y_1 = max( y_1 + width / 2 * ( 1 - box_scale ),             0 ) 
        y_2 = min( y_2 - width / 2 * ( 1 - box_scale ), inp_img.size[1]-1 )

        # Repopulate dictionary
        fixed_rect_dict_list.append(
                                    {
                                        'x1':int(x_1),
                                        'x2':int(x_2),
                                        'y1':int(y_1),
                                        'y2':int(y_2),
                                    }
                                   )
    return fixed_rect_dict_list


def plot_img_bbox(
                    inp_file,
                    bb_scale=1.0,
                    save_fig=None,
                 ):
    img = Image.open( inp_file )
    fig, ax = plt.subplots()
    ax.imshow( img )
    rect_list = get_img_bbox( file_name, img, box_scale=bb_scale )
    for rect in rect_list:
        x_1 = rect['x1']
        x_2 = rect['x2']
        y_1 = rect['y1']
        y_2 = rect['y2']
        width = x_2 - x_1
        height= y_2 - y_1
        ax.add_patch(Rectangle( 
                        (int(x_1),int(y_1)), 
                        int(width), 
                        int(height), fill=False, color='r' ) )
    if ( save_fig==None ):
        plt.show()
    else:
        plt.savefig( save_fig )
        
# Need to crop legos out of images
def crop_legos(
                inp_file_name,
                box_scale=1.0
              ):
    
    # Open the image
    inp_img = Image.open( inp_file_name )
    
    # Get the bounding boxes
    box_dict_list = get_img_bbox( inp_file_name, inp_img, box_scale=box_scale )

    # Turn into np array
    img_arr = efg.arr_from_pil( inp_img )

    # Put all the lego images in here
    img_list = []
    
    # For each box/lego found, take it
    #  out of the array
    for box in box_dict_list:
        
        # Save to a list of images
        # Images xy, np yx
        img_list.append(
                        img_arr[
                                box["y1"]:box["y2"],
                                box["x1"]:box["x2"],
                               ]
                       )
    return img_list

def generate_prediction_string(
                                inp_file_name
                              ):
    
    cropped_arr_list = crop_legos( inp_file_name, box_scale=1.1 )
    
    block_list = []
    
    for img_arr in cropped_arr_list:
        rc_ratio, row_avg, col_avg = efg.get_img_edge_data( 
                                                            img_arr, 
                                                            edge_cutoff=50,
                                                          )
    
        feature_arr = generate_feature_array( rc_ratio, row_avg, col_avg )
        
        height_str = get_height_predict( feature_arr )[0].split('_')[1]
        long_str   = get_short_predict ( feature_arr )[0].split('_')[1]
        short_str  = get_long_predict  ( feature_arr )[0].split('_')[1]
        
        full_str = short_str+'x'+long_str+' '+height_str
        
        block_list.append( full_str )
    return block_list