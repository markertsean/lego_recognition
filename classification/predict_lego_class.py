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

import generate_image_lists as giList
import generate_image_labels as giLabels
import edge_feature_generation as efg


with open('data/height_logistic_clf.pkl','r') as f:
    _LOGISTIC_HEIGHT_CLF = pkl.load( f )
with open('data/short_logistic_clf.pkl','r') as f:
    _LOGISTIC_SHORT_CLF  = pkl.load( f )
with open('data/long_logistic_clf.pkl','r') as f:
    _LOGISTIC_LONG_CLF   = pkl.load( f )
with open('data/row_col_pca.pkl'      ,'r') as f:
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