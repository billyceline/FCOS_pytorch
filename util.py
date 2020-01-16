"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import random
from crop_bbox import *

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a



##revised version for data augmentation
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True,crop = True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    #image = Image.open(line[0])
    image = Image.open(line[0]+' '+line[1])
    h, w = input_shape
    iw, ih = image.size
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
    #box = np.array([np.array(list(map(int,box.split(',')))) for box in annotation_line])
#######random crop boxes#############################
    if crop:
        box ,crop_offset = random_crop_with_constraints(box, (iw,ih), min_scale=0.5, max_scale=1,max_aspect_ratio=2, constraints=None,max_trial=50)
        image = image.crop([crop_offset[0],crop_offset[1],crop_offset[0]+crop_offset[2],crop_offset[1]+crop_offset[3]])
        iw, ih = image.size
    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box
        reorder_box = np.zeros_like(box_data[...,0:4])
        cls_data = np.zeros_like(box_data[...,4])
        #box_data[:,0:4]/=300
        reorder_box[:,0] = box_data[:,1]/input_shape[0]
        reorder_box[:,1] = box_data[:,0]/input_shape[1]
        reorder_box[:,2] = box_data[:,3]/input_shape[0]
        reorder_box[:,3] = box_data[:,2]/input_shape[1]
        cls_data = box_data[:,4]
        return image_data, reorder_box,cls_data

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    if proc_img:
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)


    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image_data)/255)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        if len(box)>max_boxes: box = box[:max_boxes]
        box[:, [0,2]] = box[:, [0,2]]*scale + dx
        box[:, [1,3]] = box[:, [1,3]]*scale + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box_data[:len(box)] = box
    reorder_box = np.zeros_like(box_data[...,0:4])
    cls_data = np.zeros_like(box_data[...,4])
    #box_data[:,0:4]/=300
    reorder_box[:,0] = box_data[:,1]/input_shape[0]
    reorder_box[:,1] = box_data[:,0]/input_shape[1]
    reorder_box[:,2] = box_data[:,3]/input_shape[0]
    reorder_box[:,3] = box_data[:,2]/input_shape[1]
    cls_data = box_data[:,4]
    return image_data, reorder_box,cls_data

def get_path_and_annotation(file_path):
    annotation = []
    img_path = []
    line_list = []
    with open(file_path,'r') as f:
        for line in f:
            temp=[]
            line = line.strip('\n')
            line_list.append(line)
        random.shuffle(line_list)
    return line_list

def read_data(annotation_list,batch_size,input_shape=(300,300),is_random=True,is_crop=True):
    #annotation_list: just the whole line from the txt file
    #return box_batch: [xmin,ymin,xmax,ymax]
    num_batch = len(annotation_list)/batch_size
    count=0
    while(True):
        image_batch = list()
        box_batch = list()
        cls_batch = list()
        for i in range(batch_size):
            temp_index = i+count*batch_size
            temp_index %=len(annotation_list) 
            image_data, box_data,cls_data = get_random_data(annotation_list[temp_index], input_shape, random=is_random, max_boxes=30, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True,crop=is_crop) 
            image_batch.append(image_data)
            box_batch.append(box_data)
            cls_batch.append(cls_data)
        count+=1
        image_batch = np.array(image_batch)
        box_batch = np.array(box_batch)
        cls_batch = np.array(cls_batch)
        yield image_batch,box_batch,cls_batch

def read_one_img(image_path,input_shape):
    #read one img and use keep aspect ratio resize
    image = Image.open(image_path)
    iw, ih = image.size
    h, w = input_shape
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2
    image_data=0
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image)/255.
    return image_data

def get_gt_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    #image = Image.open(line[0])
    image = Image.open(line[0]+' '+line[1])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
    #box = np.array([np.array(list(map(int,box.split(',')))) for box in annotation_line])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box
        box_data[:,0:4]/=300
        reorder_box = np.zeros_like(box_data)
        #cls_data = np.zeros_like(box_data[...,4])
        reorder_box[:,0] = box_data[:,1]
        reorder_box[:,1] = box_data[:,0]
        reorder_box[:,2] = box_data[:,3]
        reorder_box[:,3] = box_data[:,2]
        reorder_box[:,4] = box_data[:,4]
        return (line[0]+' '+line[1]),image_data,reorder_box

def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    return: IoU matrix: shape: [N, V]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou

#def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
#     '''random preprocessing for real-time data augmentation'''
#     line = annotation_line.split()
#     #image = Image.open(line[0])
#     image = Image.open(line[0]+' '+line[1])
#     iw, ih = image.size
#     h, w = input_shape
#     box = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
#     #box = np.array([np.array(list(map(int,box.split(',')))) for box in annotation_line])

#     if not random:
#         # resize image
#         scale = min(w/iw, h/ih)
#         nw = int(iw*scale)
#         nh = int(ih*scale)
#         dx = (w-nw)//2
#         dy = (h-nh)//2
#         image_data=0
#         if proc_img:
#             image = image.resize((nw,nh), Image.BICUBIC)
#             new_image = Image.new('RGB', (w,h), (128,128,128))
#             new_image.paste(image, (dx, dy))
#             image_data = np.array(new_image)/255.

#         # correct boxes
#         box_data = np.zeros((max_boxes,5))
#         if len(box)>0:
#             np.random.shuffle(box)
#             if len(box)>max_boxes: box = box[:max_boxes]
#             box[:, [0,2]] = box[:, [0,2]]*scale + dx
#             box[:, [1,3]] = box[:, [1,3]]*scale + dy
#             box_data[:len(box)] = box
#         reorder_box = np.zeros_like(box_data[...,0:4])
#         cls_data = np.zeros_like(box_data[...,4])
#         #box_data[:,0:4]/=300
#         reorder_box[:,0] = box_data[:,1]
#         reorder_box[:,1] = box_data[:,0]
#         reorder_box[:,2] = box_data[:,3]
#         reorder_box[:,3] = box_data[:,2]
#         cls_data = box_data[:,4]
#         return image_data, reorder_box,cls_data

#     # resize image
#     new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
#     scale = rand(.25, 2)
#     if new_ar < 1:
#         nh = int(scale*h)
#         nw = int(nh*new_ar)
#     else:
#         nw = int(scale*w)
#         nh = int(nw/new_ar)
#     image = image.resize((nw,nh), Image.BICUBIC)

#     # place image
#     dx = int(rand(0, w-nw))
#     dy = int(rand(0, h-nh))
#     new_image = Image.new('RGB', (w,h), (128,128,128))
#     new_image.paste(image, (dx, dy))
#     image = new_image

#     # flip image or not
#     flip = rand()<.5
#     if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

#     # distort image
#     hue = rand(-hue, hue)
#     sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
#     val = rand(1, val) if rand()<.5 else 1/rand(1, val)
#     x = rgb_to_hsv(np.array(image)/255.)
#     x[..., 0] += hue
#     x[..., 0][x[..., 0]>1] -= 1
#     x[..., 0][x[..., 0]<0] += 1
#     x[..., 1] *= sat
#     x[..., 2] *= val
#     x[x>1] = 1
#     x[x<0] = 0
#     image_data = hsv_to_rgb(x) # numpy array, 0 to 1

#     # correct boxes
#     box_data = np.zeros((max_boxes,5))
#     if len(box)>0:
#         np.random.shuffle(box)
#         box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
#         box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
#         if flip: box[:, [0,2]] = w - box[:, [2,0]]
#         box[:, 0:2][box[:, 0:2]<0] = 0
#         box[:, 2][box[:, 2]>w] = w
#         box[:, 3][box[:, 3]>h] = h
#         box_w = box[:, 2] - box[:, 0]
#         box_h = box[:, 3] - box[:, 1]
#         box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
#         if len(box)>max_boxes: box = box[:max_boxes]
#         box_data[:len(box)] = box
# #change annotation format from xmin,ymin,xmax,ymax to ymin,xmin,ymax,xmax
#     reorder_box = np.zeros_like(box_data[...,0:4])
#     #box_data[:,0:4]/=300
#     cls_data = np.zeros_like(box_data[...,4])
#     reorder_box[:,0] = box_data[:,1]
#     reorder_box[:,1] = box_data[:,0]
#     reorder_box[:,2] = box_data[:,3]
#     reorder_box[:,3] = box_data[:,2]
#     cls_data = box_data[:,4]
#     return image_data, reorder_box,cls_data

