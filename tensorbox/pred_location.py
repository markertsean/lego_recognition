"""
This file is designed for prediction of bounding boxes for a single image.

Predictions could be made in two ways: command line style or service style. Command line style denotes that one can 
run this script from the command line and configure all options right in the command line. Service style allows 
to call :func:`initialize` function once and call :func:`hot_predict` function as many times as it needed to. 

"""

import tensorflow as tf
import os, json, subprocess, random
from optparse import OptionParser
from os import path

from scipy.misc import imread, imresize, imsave
import numpy as np
from PIL import Image, ImageDraw

from train import build_forward
#from utils.annolist import AnnotationLib as al
#from utils.train_utils import add_rectangles, rescale_boxes
#from utils.data_utils import Rotate90
#from annolist import AnnotationLib as al
import AnnotationLib as al
from train_utils import add_rectangles, rescale_boxes
from data_utils import Rotate90

if __package__ is None:
    import sys
    sys.path.append(path.abspath(path.join(path.dirname(__file__), path.pardir, 'detect-widgets/additional')))


def initialize(weights_path, hypes_path, options=None):
    """Initialize prediction process.

    All long running operations like TensorFlow session start and weights loading are made here.

    Args:
        weights_path (string): The path to the model weights file. 
        hypes_path (string): The path to the hyperparameters file. 
        options (dict): The options dictionary with parameters for the initialization process.

    Returns (dict):
        The dict object which contains `sess` - TensorFlow session, `pred_boxes` - predicted boxes Tensor, 
          `pred_confidences` - predicted confidences Tensor, `x_in` - input image Tensor, 
          `hypes` - hyperparametets dictionary.
    """

    H = prepare_options(hypes_path, options)

    
    if H is None:
        return None

    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas \
            = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(
            tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
            [grid_area, H['rnn_len'], H['num_classes']])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, weights_path)
    return {'sess': sess, 'pred_boxes': pred_boxes, 'pred_confidences': pred_confidences, 'x_in': x_in, 'hypes': H}


def hot_predict(image_path, parameters, to_json=True, verbose=False):
    """Makes predictions when all long running preparation operations are made.

    Args:
        image_path (string): The path to the source image.
        parameters (dict): The parameters produced by :func:`initialize`.

    Returns (Annotation):
        The annotation for the source image.
    """

    H = parameters.get('hypes', None)
    if H is None:
        return None

    # The default options for prediction of bounding boxes.
    options = H['evaluate']
    if 'pred_options' in parameters:
        # The new options for prediction of bounding boxes
        for key, val in parameters['pred_options'].items():
            options[key] = val

    # predict
    use_sliding_window = H.get('sliding_predict', {'enable': False}).get('enable', False)
    
    if use_sliding_window:
        if verbose:
            print('Sliding window mode on')
        return sliding_predict(image_path, parameters, to_json, H, options)
    else:
        if verbose:
            print('Sliding window mode off')
        return regular_predict(image_path, parameters, to_json, H, options)


def calculate_medium_box(boxes):
    conf_sum = reduce(lambda t, b: t + b.score, boxes, 0)
    aggregation = {}
    for name in ['x1', 'y1', 'x2', 'y2']:
        aggregation[name] = reduce(lambda t, b: t+b.__dict__[name]*b.score, boxes, 0) / conf_sum

    new_box = al.AnnoRect(**aggregation)
    new_box.classID = boxes[0].classID
    new_box.score = conf_sum / len(boxes)
    return new_box


def non_maximum_suppression(boxes):
    conf = [box.score for box in boxes]
    ind = np.argmax(conf)
    if isinstance(ind, int):
        return boxes[ind]
    else:
        random.seed()
        num = random.randint(0, len(ind))
        return boxes[num]


def combine_boxes(boxes, iou_min, nms, verbose=False):
    neighbours, result = [], []
    for i, box in enumerate(boxes):
        cur_set = set()
        cur_set.add(i)
        for j, neigh_box in enumerate(boxes):
            iou_val = box.iou(neigh_box)
            if verbose:
                print(i, j, iou_val )
            if i != j and iou_val > iou_min:
                cur_set.add(j)

        if len(cur_set) == 0:
            result.append(box)
        else:
            for group in neighbours:
                if len(cur_set.intersection(group)) > 0:
                    neighbours.remove(group)
                    cur_set = cur_set.union(group)
            neighbours.append(cur_set)

    for group in neighbours:
        cur_boxes = [boxes[i] for i in group]
        if nms:
            medium_box = non_maximum_suppression(cur_boxes)
        else:
            medium_box = calculate_medium_box(cur_boxes)
        result.append(medium_box)

    return result


def shift_boxes(pred_anno_rects, margin):
    for box in pred_anno_rects:
        box.y1 += margin
        box.y2 += margin


def to_box(anno_rect, parameters):
    box = {}
    box['x1'] = anno_rect.x1
    box['x2'] = anno_rect.x2
    box['y1'] = anno_rect.y1
    box['y2'] = anno_rect.y2
    box['score'] = anno_rect.score
    if 'classID' in parameters:
        box['classID'] = parameters['classID']
    else:
        box['classID'] = anno_rect.classID
    return box


def regular_predict(image_path, parameters, to_json, H, options):
    orig_img = imread(image_path)[:, :, :3]
    img = Rotate90.do(orig_img)[0] if 'rotate90' in H['data'] and H['data']['rotate90'] else orig_img
    img = imresize(img, (H['image_height'], H['image_width']), interp='cubic')
    np_pred_boxes, np_pred_confidences = parameters['sess']. \
        run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})
    
    image_info = {'path': image_path, 'original_shape': img.shape[:2], 'transformed': img}
    pred_anno = postprocess_regular(image_info, np_pred_boxes, np_pred_confidences, H, options)
    result = [r.writeJSON() for r in pred_anno] if to_json else pred_anno

    ret_list = [ {
                  'x1':r.x1,
                  'x2':r.x2,
                  'y1':r.y1,
                  'y2':r.y2,        
    } for r in pred_anno]
    print ret_list
    
    return result


def propose_slides(img_height, slide_height, slide_overlap):
    slides = []
    step = slide_height - slide_overlap
    for top in range(0, img_height - slide_height, step):
        slides.append((top, top+slide_height))
    # there is some space left which was not covered by slides; make slide at the bottom of image
    slides.append((img_height-slide_height, img_height))
    return slides


def sliding_predict(image_path, parameters, to_json, H, options):
    orig_img = imread(image_path)[:, :, :3]
    height, width, _ = orig_img.shape
    if options.get('verbose', False):
        print(width, height)

    sl_win_options = H['sliding_predict']
    assert (sl_win_options['window_height'] > sl_win_options['overlap'])
    slides = propose_slides(height, sl_win_options['window_height'], sl_win_options['overlap'])

    result = []
    for top, bottom in slides:
        bottom = min(height, top + sl_win_options['window_height'])
        if options.get('verbose', False):
            print('Slide: ', 0, top, width, bottom)

        img = orig_img[top:bottom, 0:width]
        img = Rotate90.do(img)[0] if 'rotate90' in H['data'] and H['data']['rotate90'] else img
        img = imresize(img, (H['image_height'], H['image_width']), interp='cubic')

        np_pred_boxes, np_pred_confidences = parameters['sess']. \
            run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})
        image_info = {'path': image_path, 'original_shape': (bottom-top, width), 'transformed': img, 'a': orig_img[top:bottom, 0:width]}

        pred_boxes = postprocess_regular(image_info, np_pred_boxes, np_pred_confidences, H, options)
        shift_boxes(pred_boxes, top)
        result.extend(pred_boxes)

    result = combine_boxes(result, sl_win_options['iou_min'], sl_win_options['nms'])
    result = [r.writeJSON() for r in result] if to_json else result
    return result


def postprocess_regular(image_info, np_pred_boxes, np_pred_confidences, H, options):
    pred_anno = al.Annotation()
    pred_anno.imageName = image_info['path']
    pred_anno.imagePath = os.path.abspath(image_info['path'])
    _, rects = add_rectangles(H, [image_info['transformed']], np_pred_confidences, np_pred_boxes, use_stitching=True,
                              rnn_len=H['rnn_len'], min_conf=options['min_conf'], tau=options['tau'],
                              show_suppressed=False)

    h, w = image_info['original_shape']
    if 'rotate90' in H['data'] and H['data']['rotate90']:
        # original image height is a width for rotated one
        rects = Rotate90.invert(h, rects)

    rects = [r for r in rects if r.x1 < r.x2 and r.y1 < r.y2 and r.score > options['min_conf']]
    pred_anno.rects = rects
    pred_anno = rescale_boxes((H['image_height'], H['image_width']), pred_anno, h, w)
    return pred_anno


def prepare_options(hypes_path='hypes.json', options=None):
    """Sets parameters of the prediction process. If evaluate options provided partially, it'll merge them.
    The priority is given to options argument to overwrite the same obtained from the hyperparameters file.

    Args:
        hypes_path (string): The path to model hyperparameters file.
        options (dict): The command line options to set before start predictions.

    Returns (dict):
        The model hyperparameters dictionary.
    """

    with open(hypes_path, 'r') as f:
        H = json.load(f)


    # set default options values if they were not provided
    if options is None:
        if 'evaluate' in H:
            options = H['evaluate']
        else:
            print ('Evaluate parameters were not found! You can provide them through hyperparameters json file '
                   'or hot_predict options parameter.')
            return None
    else:
        if 'evaluate' not in H:
            H['evaluate'] = {}
        # merge options argument into evaluate options from hyperparameters file
#        for key, val in options.items():
        H['evaluate']['tau'] = 0.25
        H['evaluate']['gpu'] = 0
        H['evaluate']['min_conf'] = 0.2

    if H['evaluate'].get('gpu', False):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(H['evaluate']['gpu'])
    return H


def save_results(image_path, anno, fname='result'):
    """Saves results of the prediction.

    Args:
        image_path (string): The path to source image to predict bounding boxes.
        anno (Annotation, list): The predicted annotations for source image or the list of bounding boxes.

    Returns:
        Nothing.
    """

    # draw
    new_img = Image.open(image_path)
    d = ImageDraw.Draw(new_img)
    is_list = type(anno) is list
    rects = anno if is_list else anno.rects
    for r in rects:
        if is_list:
            d.rectangle([r['x1'], r['y1'], r['x2'], r['y2']], outline=(255, 0, 0))
        else:
            d.rectangle([r.left(), r.top(), r.right(), r.bottom()], outline=(255, 0, 0))

    # save
    prediction_image_path = os.path.join(os.path.dirname(image_path), fname + '.png')
    new_img.save(prediction_image_path)
    subprocess.call(['chmod', '644', prediction_image_path])

    fpath = os.path.join(os.path.dirname(image_path), fname + '.json')
    if is_list:
        json.dump({'image_path': prediction_image_path, 'rects': anno}, open(fpath, 'w'))
    else:
        al.saveJSON(fpath, anno)
    subprocess.call(['chmod', '644', fpath])


    
    
    
def no_longer_main():
    parser = OptionParser(usage='usage: %prog [options] <image> <hypes>')
    parser.add_option('--gpu', action='store', type='int', default=0)
    parser.add_option('--tau', action='store', type='float', default=0.25)
    parser.add_option('--min_conf', action='store', type='float', default=0.2)

    (options, args) = parser.parse_args()
    if len(args) < 2:
        print ('Provide path configuration json file')
        return

    image_filename = args[0]
    hypes_path = args[1]
    config = json.load(open(hypes_path, 'r'))
#    weights_path = os.path.join(os.path.dirname(hypes_path), config['solver']['weights'])
    weights_path = 'lego_train/overfeat_rezoom_2018_06_14_09.20/save.ckpt-1500'
    init_params = initialize(weights_path, hypes_path, options)
    init_params['pred_options'] = {'verbose': True}
    pred_anno = hot_predict(image_filename, init_params)
    save_results(image_filename, pred_anno, 'predictions_sliced')

    
    
    
    

    
    
    
    
    
    
    
    
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)

#_weights_full_path = '/home/sean/Insight/other_tensorbox/TensorBox/lego_train/overfeat_rezoom_2018_06_14_09.20/save.ckpt-1500'
_weights_full_path = '/home/sean/Insight/other_tensorbox/TensorBox/lego_dirty/overfeat_rezoom_2018_06_17_23.32/save.ckpt-15000'
_hypes_full_path   = '/home/sean/Insight/legos/tensorbox/hypes/hypes.json'


def rect_predict(image_path, parameters, to_json, H, options):
    orig_img = imread(image_path)[:, :, :3]
    img = Rotate90.do(orig_img)[0] if 'rotate90' in H['data'] and H['data']['rotate90'] else orig_img
    img = imresize(img, (H['image_height'], H['image_width']), interp='cubic')
    np_pred_boxes, np_pred_confidences = parameters['sess']. \
        run([parameters['pred_boxes'], parameters['pred_confidences']], feed_dict={parameters['x_in']: img})
    
    image_info = {'path': image_path, 'original_shape': img.shape[:2], 'transformed': img}
    pred_anno = postprocess_regular(image_info, np_pred_boxes, np_pred_confidences, H, options)

    ret_list = [ {
                  'x1':r.x1,
                  'x2':r.x2,
                  'y1':r.y1,
                  'y2':r.y2,        
    } for r in pred_anno]
    
    return ret_list



def pred_lego_locations( inp_image_path ):
    
    image_filename = inp_image_path
    hypes_path     = _hypes_full_path
    weights_path   = _weights_full_path



    # Set up a lot of parameters
    config = json.load(open(hypes_path, 'r'))
    
    options = ''
    
    init_params = initialize(weights_path, hypes_path, options)
    init_params['pred_options'] = {'verbose': True}

    H = init_params.get('hypes', None)
    if H is None:
        return None
    # The default options for prediction of bounding boxes.
    options = H['evaluate']
    if 'pred_options' in init_params:
        # The new options for prediction of bounding boxes
        for key, val in init_params['pred_options'].items():
            options[key] = val

            
    # Do the predicty things
    return rect_predict(image_filename, init_params, False, H, options)

    
    
    

    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()
