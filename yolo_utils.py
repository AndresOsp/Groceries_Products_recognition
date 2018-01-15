import colorsys
import imghdr
import os
import random
from keras import backend as K
from matplotlib.pyplot import imshow

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from yad2k.models.keras_yolo import preprocess_true_boxes, tiny_yolo_body, yolo_eval, yolo_head, yolo_loss

#########################################################
def read_images(images_path,width=416,height=416):
    '''
    loads and preprocess the images from the indicated pads in the txt
    '''
    with open(images_path) as f:
        images_path_s = f.readlines()
    images_path_s = [c.strip() for c in images_path_s]
    
    return preprocess_images(images_path_s, width,height)

#########################################################
def preprocess_images(img_paths, width=416,height=416):
    images = [Image.open(i) for i in img_paths]
    
    processed_images = [i.resize((width, height), Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]
    
    return np.asarray(processed_images), images


#########################################################
def preprocess_image_2(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


#########################################################
def read_boxes(boxes_path):
    '''loads the images from the indicated pads in the txt'''
    
    # read the file 
    with open(boxes_path) as f:
        boxes_path_s = f.readlines()
    boxes_path_s = [c.strip() for c in boxes_path_s]
    
    # create array with the boxes
    boxes=[]
    for path_b in boxes_path_s:
        temp=np.loadtxt(path_b)
        boxes.append(temp)
        
    # find the max number of boxes
    max_boxes = 0
    for boxz in boxes:
        if boxz.shape[0] > max_boxes:
            max_boxes = boxz.shape[0]
            
    # add zero pad for training
    for i, boxz in enumerate(boxes):
        if boxz.shape[0]  < max_boxes:
            zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
            boxes[i] = np.vstack((boxz, zero_padding))

    return np.asarray(boxes)

#########################################################
def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#########################################################
def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

#########################################################
def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

#########################################################
def yolo_boxes_to_corners(box,image_shape):
    """Convert YOLO box predictions to bounding box corners."""

    detected_boxes=np.sum(box[:,0]>0)
    
    box=box[0:detected_boxes,0:4]
    
    box_corner=np.zeros_like(box)
    
    dw = image_shape[1]
    dh = image_shape[0]
    
    box_corner[:,1]=(dh*(box[:,0]-(box[:,2]/2)))
    box_corner[:,0]=(dw*(box[:,1]-(box[:,3]/2)))
    box_corner[:,3]=(dh*(box[:,0]+(box[:,2]/2)))
    box_corner[:,2]=(dw*(box[:,1]+(box[:,3]/2)))
    
    return box_corner

#########################################################
def draw_boxes_simple(image, boxes):
    
    print(image.size)
    boxes = yolo_boxes_to_corners(boxes,image.size)
    thickness = (image.size[0] + image.size[1]) // 300

    for i, box in (enumerate(boxes)):

        draw = ImageDraw.Draw(image)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i])
            
        imshow(image)

        del draw
        
#########################################################
def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

#########################################################
def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

#########################################################
def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes
