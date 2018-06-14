import os
import json
import subprocess

from tensorbox_model import TensorBox

_json_format = '[\n{\n"rects": [\n  {\n   "y1": 1729.0,\n   "x2":  963.0,\n   "x1":  184.0,\n   "y2": 2508.0\n  }\n ],\n "image_path": "%s"\n}\n]\n'

def _generate_json( inp_file_path ):
    
    with open('sample.json','w') as f:
        f.write( _json_format % inp_file_path )
        
def get_rect( inp_file_name ):

    _generate_json( inp_file_name )
    
    _weights    = 'nn_model/save.ckpt-5000'
    _test_boxes = '/home/sean/Desktop/lego_images_bounding_box/short.json'#'sample.json'
    _min_conf   = 0.2
    _tau        = 0.25
    _show_supp  = True
    _iou_thresh = 0.5
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    hypes_file = '%s/hypes.json' % os.path.dirname( _weights )
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = ''
    pred_boxes = '%s.%s%s' % ( _weights, expname, os.path.basename( _test_boxes))
    true_boxes = '%s.gt_%s%s' % ( _weights, expname, os.path.basename( _test_boxes))

    
    tensorbox = TensorBox(H)
    rect_list = tensorbox.pred( _weights,
                                                   _test_boxes,
                                                   _min_conf,
                                                   _tau,
                                                   _show_supp,
                                                  expname)

    print rect_list

def main():

    get_rect('foo')
    sys.exit()
    _weights    = 'nn_model/save.ckpt-5000'
    _test_boxes = '/home/sean/Desktop/lego_images_bounding_box/short.json'
    _min_conf   = 0.2
    _tau        = 0.25
    _show_supp  = True
    _iou_thresh = 0.5
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    hypes_file = '%s/hypes.json' % os.path.dirname( _weights )
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = ''
    pred_boxes = '%s.%s%s' % ( _weights, expname, os.path.basename( _test_boxes))
    true_boxes = '%s.gt_%s%s' % ( _weights, expname, os.path.basename( _test_boxes))

    
    tensorbox = TensorBox(H)
    rect_list = tensorbox.pred( _weights,
                                                   _test_boxes,
                                                   _min_conf,
                                                   _tau,
                                                   _show_supp,
                                                  expname)

    print rect_list

if __name__ == '__main__':
    main()
