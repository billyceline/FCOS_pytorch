import numpy as np
from flags_and_variables import *
import torch
import _C
nms = _C.nms
###get ymin ,xmin ,ymax,xmax seperatelty

def encode_boxes(annotation_batch,cls_batch,feature_size,stride,concatenate=True):
    m = np.array([-1,64,128,256,512,1e8])
    ymin_true = np.expand_dims(annotation_batch[...,0],axis=2)#(b,N,1)
    ymax_true = np.expand_dims(annotation_batch[...,2],axis=2)
    xmin_true = np.expand_dims(annotation_batch[...,1],axis=2)
    xmax_true = np.expand_dims(annotation_batch[...,3],axis=2)

    l_t_r_b_true = []
    x_list=[]
    y_list=[]
    matched_true_boxes = []
    matched_true_classes = []
    matched_true_centerness = []
    for feature_index in range(5):
        m_min =m[feature_index]
        m_max =m[feature_index+1]
        offset = np.math.floor(stride[feature_index]/2)
        y_center_mapping = np.array([(i*stride[feature_index]+offset) for i in range(feature_size[feature_index][0])])/image_height #[w]
        x_center_mapping = np.array([(i*stride[feature_index]+offset) for i in range(feature_size[feature_index][1])])/image_width#[H]
#         x_list.append(x_center_mapping)
#         y_list.append(y_center_mapping)

    ####whether center in box
        y_mask = np.logical_and((y_center_mapping>ymin_true),(y_center_mapping<ymax_true))#[b,N,W]
        x_mask = np.logical_and((x_center_mapping>xmin_true),(x_center_mapping<xmax_true))#[b,N,H]
        y_mask_f = y_mask.astype(np.float32)
        x_mask_f = x_mask.astype(np.float32)

    #calculate top,bottom,left,right and concate together  
        t_true = (y_center_mapping - ymin_true)*y_mask_f
        #t_true = np.expand_dims(np.expand_dims(t_true,3),axis=-1) * np.ones([y_mask.shape[0],y_mask.shape[1],y_mask.shape[2],x_mask.shape[2],1])
        t_true = np.tile(np.expand_dims(np.expand_dims(t_true,3),axis=-1), [1,1,1,x_mask.shape[2],1]) #(6, 30, 100, 128, 1)

        b_true = (ymax_true - y_center_mapping)*y_mask_f
        b_true = np.tile(np.expand_dims(np.expand_dims(b_true,3),axis=-1), [1,1,1,x_mask.shape[2],1])#(6, 30, 100, 128, 1)

        l_true = (x_center_mapping - xmin_true)*x_mask_f
        l_true = np.tile(np.expand_dims(np.expand_dims(l_true,2),axis=-1),[1,1,y_mask.shape[2],1,1])#(6, 30, 100, 128, 1)

        r_true = (xmax_true - x_center_mapping)*x_mask_f
        r_true = np.tile(np.expand_dims(np.expand_dims(r_true,2),axis=-1),[1,1,y_mask.shape[2],1,1])#(6, 30, 100, 128, 1)

        tblr = np.concatenate([t_true,b_true,l_true,r_true],axis=-1)
        tblr_temp = np.concatenate([t_true*image_height,b_true*image_height,l_true*image_width,r_true*image_width],axis=-1)#(6, 30, 100, 128, 4)

        tblr_mask = np.expand_dims(np.logical_and((np.max(tblr_temp,axis=-1)>m_min),(np.max(tblr_temp,axis=-1)<=m_max)),axis=-1)#(b,N,W,H,1)
        xy_mask = (np.logical_and(np.expand_dims(y_mask,axis=3),np.expand_dims(x_mask,axis=2))).astype(np.float32)
        xy_mask = np.expand_dims(xy_mask,axis=-1)#(?,?,100,128,1)

        true_boxes = np.expand_dims(np.expand_dims(annotation_batch,axis=2),axis=3)#(?,?,1,1,4)
        true_classes = np.expand_dims(np.expand_dims(cls_batch,axis=2),axis=3)
        #encode true boxes to feature map
        true_boxes = true_boxes * xy_mask * tblr_mask#(?,?,100,128,4)
        true_classes = true_classes * xy_mask * tblr_mask
        tblr = tblr *xy_mask *tblr_mask

        area = (true_boxes[...,2]-true_boxes[...,0])*(true_boxes[...,3]-true_boxes[...,1])
        area[area==0] = 1000000 #exclude 0
        index_min = np.argmin(area,axis=1)
        #get the corresponding minimal area encoded boxes
        pos_true_boxes=[]
        pos_true_classes=[]
        for b in range(batch_size):
            w_boxes=[]
            w_classes=[]
            for w in range(feature_size[feature_index][0]):
                h_boxes = []
                h_classes=[]
                for h in range(feature_size[feature_index][1]):
                    h_boxes.append(tblr[b,:,w,h,:][index_min[b,w,h]])
                    h_classes.append(true_classes[b,:,w,h,:][index_min[b,w,h]])
                w_boxes.append(h_boxes)
                w_classes.append(h_classes)
            pos_true_boxes.append(w_boxes)
            pos_true_classes.append(w_classes)
        pos_true_boxes = np.array(pos_true_boxes)
        pos_true_classes = np.squeeze(np.array(pos_true_classes)).astype(np.int32)
        ###calculate centerness
        tb_max = np.max(pos_true_boxes[...,0:2],axis=-1)
        tb_min = np.min(pos_true_boxes[...,0:2],axis=-1)
        lr_max = np.max(pos_true_boxes[...,2:],axis=-1)
        lr_min = np.min(pos_true_boxes[...,2:],axis=-1)
        centerness = np.expand_dims(np.sqrt((lr_min/(lr_max+1e-8)) * (tb_min/(tb_max+1e-8))),axis=-1)
        matched_true_boxes.append(np.reshape(pos_true_boxes,(-1,feature_size[feature_index][0]*feature_size[feature_index][1],4)))
        matched_true_classes.append(np.reshape(pos_true_classes,(-1,feature_size[feature_index][0]*feature_size[feature_index][1],num_classes)))
        matched_true_centerness.append(np.reshape(centerness,(-1,feature_size[feature_index][0]*feature_size[feature_index][1],1)))
    if(concatenate == True):
        matched_true_boxes = np.concatenate(matched_true_boxes,axis=1)
        matched_true_classes = np.concatenate(matched_true_classes,axis=1)
        matched_true_centerness = np.concatenate(matched_true_centerness,axis=1)
    return matched_true_boxes,matched_true_classes,matched_true_centerness

def predict_outputs(centerness_pred,classes_pred,localization_pred,feature_size,stride,inference_threshold=0.2):
    
    m = np.array([-1,64,128,256,512,np.inf])
    center_list = []
    m_min_list = []
    m_max_list = []
    for i in range(5):
        #change top,bottom,left,right to ymin,xmin,ymax,xmax
    # feature_size=[(100,128),(50,64),(25,32),(13,16),(7,8)]
    # stride=[8,16,32,64,128]
        m_min =m[i]
        m_max =m[i+1]
        #ensure the predicted boxes max(top,bottom,left,right) is in the domain(m_h_min,m_w_max)
            #the mim(top,bottom,left,right) is bigger than 1 pixel
        offset = np.math.floor(stride[i]/2)
        y_center_mapping = np.array([(j*stride[i]+offset) for j in range(feature_size[i][0])])
        x_center_mapping = np.array([(j*stride[i]+offset) for j in range(feature_size[i][1])])
        y_center_mapping = np.expand_dims(np.tile(np.expand_dims(y_center_mapping,axis=-1),[1,feature_size[i][1]]),axis=-1)
        x_center_mapping = np.expand_dims(np.tile(np.expand_dims(x_center_mapping,axis=0),[feature_size[i][0],1]),axis=-1)
        center = np.concatenate([y_center_mapping,x_center_mapping],axis=-1).reshape(-1,(feature_size[i][0]*feature_size[i][1]),2)
        center_list.append(center)
        m_min = np.ones_like(y_center_mapping)*m_min
        m_max = np.ones_like(x_center_mapping)*m_max
        m_min_list.append(m_min.reshape(-1,(feature_size[i][0]*feature_size[i][1])))
        m_max_list.append(m_max.reshape(-1,(feature_size[i][0]*feature_size[i][1])))
    center_list = np.concatenate(center_list,axis=1) #(1, 17064, 2)
    m_min_list = np.concatenate(m_min_list,axis=1)#(1, 17064)
    m_max_list = np.concatenate(m_max_list,axis=1)#(1, 17064)
    
    center_list_cuda = torch.from_numpy(center_list).to(device)
    m_min_list_cuda = torch.from_numpy(m_min_list).to(device)
    m_max_list_cuda = torch.from_numpy(m_max_list).to(device)

    localization_pred_cuda = localization_pred * torch.from_numpy(np.array([image_height,image_height,image_width,image_width])).to(device)
    localization_mask_1 = torch.unsqueeze(((torch.max(localization_pred_cuda,-1).values>m_min_list_cuda)*(torch.max(localization_pred_cuda,-1).values<m_max_list_cuda)),-1)
    localization_mask_2 = torch.unsqueeze((torch.min(localization_pred_cuda,-1).values>1),-1).float()
    localization_mask = localization_mask_1.float() * localization_mask_2
    localization_pred_masked = localization_pred_cuda * localization_mask
    centerness_pred_masked  = centerness_pred *localization_mask
    classes_pred_masked = classes_pred * localization_mask
    
    center_y = center_list_cuda[...,0].clone()
    center_x = center_list_cuda[...,1].clone()
    top = localization_pred_masked[...,0].clone()
    bottom = localization_pred_masked[...,1].clone()
    left = localization_pred_masked[...,2].clone()
    right = localization_pred_masked[...,3].clone()
    
    #convert from top,bottom,left,right to xmin,ymin,xmax,ymax
    ymin = torch.unsqueeze((center_y-top),-1)
    ymax = torch.unsqueeze((center_y+bottom),-1)
    xmin = torch.unsqueeze((center_x-left),-1)
    xmax = torch.unsqueeze((center_x+right),-1)
    #deal with boxes outside the region
    ymax[ymax>image_height]=image_height
    ymin[ymin>image_height]=image_height
    xmax[xmax>image_width]=image_width
    xmin[xmin>image_width]=image_width
    ymax[ymax<0]=0
    ymin[ymin<0]=0
    xmin[xmin<0]=0
    xmax[xmax<0]=0
    boxes_pred_masked = torch.cat([xmin,ymin,xmax,ymax],-1)
    centerness_pred_ = torch.squeeze(centerness_pred_masked,0)
    classes_pred_ = torch.squeeze(classes_pred_masked,0)
    localization_pred_ = torch.squeeze(boxes_pred_masked,0)
    
    score_pred = centerness_pred_*classes_pred_
#     score_pred = classes_pred
    mask = (score_pred > inference_threshold)
    
    
    _boxes = []
    _classes = []
    _scores = []
    for c in range(num_classes):
    ##nms
        _localization_pred = localization_pred_[mask[:,c]]
        _scores_pred = score_pred[:,c][mask[:,c]]
        nms_index = nms(_localization_pred, _scores_pred, 0.5)
        _boxes.append(_localization_pred[nms_index])
        _scores_pred = _scores_pred[nms_index]
        _scores.append(_scores_pred)
        _classes.append(torch.ones_like(_scores_pred) * c)
    _boxes = torch.cat(_boxes,0)
    _scores = torch.cat(_scores,0)
    _classes = torch.cat(_classes,0)
    
    return _scores,_classes,_boxes


def find_best_threshold(centerness_pred,classes_pred,localization_pred,feature_size,stride):
    m = np.array([-1,64,128,256,512,np.inf])
    center_list = []
    m_min_list = []
    m_max_list = []
    for i in range(5):
        #change top,bottom,left,right to ymin,xmin,ymax,xmax
    # feature_size=[(100,128),(50,64),(25,32),(13,16),(7,8)]
    # stride=[8,16,32,64,128]
        m_min =m[i]
        m_max =m[i+1]
        #ensure the predicted boxes max(top,bottom,left,right) is in the domain(m_h_min,m_w_max)
            #the mim(top,bottom,left,right) is bigger than 1 pixel
        offset = np.math.floor(stride[i]/2)
        y_center_mapping = np.array([(j*stride[i]+offset) for j in range(feature_size[i][0])])
        x_center_mapping = np.array([(j*stride[i]+offset) for j in range(feature_size[i][1])])
        y_center_mapping = np.expand_dims(np.tile(np.expand_dims(y_center_mapping,axis=-1),[1,feature_size[i][1]]),axis=-1)
        x_center_mapping = np.expand_dims(np.tile(np.expand_dims(x_center_mapping,axis=0),[feature_size[i][0],1]),axis=-1)
        center = np.concatenate([y_center_mapping,x_center_mapping],axis=-1).reshape(-1,(feature_size[i][0]*feature_size[i][1]),2)
        center_list.append(center)
        m_min = np.ones_like(y_center_mapping)*m_min
        m_max = np.ones_like(x_center_mapping)*m_max
        m_min_list.append(m_min.reshape(-1,(feature_size[i][0]*feature_size[i][1])))
        m_max_list.append(m_max.reshape(-1,(feature_size[i][0]*feature_size[i][1])))
    center_list = np.concatenate(center_list,axis=1) #(1, 17064, 2)
    m_min_list = np.concatenate(m_min_list,axis=1)#(1, 17064)
    m_max_list = np.concatenate(m_max_list,axis=1)#(1, 17064)

    localization_pred = localization_pred*np.array([image_height,image_height,image_width,image_width])
    localization_mask = tf.expand_dims(tf.logical_and((tf.reduce_max(localization_pred,axis=-1)>m_min_list),(tf.reduce_max(localization_pred,axis=-1)<m_max_list)),axis=-1)
    localization_mask_2 = tf.cast(tf.expand_dims((tf.reduce_min(localization_pred,axis=-1)>1),axis=-1),tf.float32)
    localization_mask = tf.cast(localization_mask,tf.float32) * localization_mask_2
    localization_pred = localization_pred * localization_mask
    centerness_pred  = centerness_pred *localization_mask
    classes_pred = classes_pred * localization_mask
    
    
    ymin = tf.expand_dims((center_list[...,0]-localization_pred[...,0]),axis=-1)
    ymax = tf.expand_dims((center_list[...,0]+localization_pred[...,1]),axis=-1)
    xmin = tf.expand_dims((center_list[...,1]-localization_pred[...,2]),axis=-1)
    xmax = tf.expand_dims((center_list[...,1]+localization_pred[...,3]),axis=-1)

    localization_pred = tf.concat([ymin,xmin,ymax,xmax],axis=-1)
    centerness_pred = tf.squeeze(centerness_pred,axis=0)
    classes_pred = tf.squeeze(classes_pred,axis=0)
    localization_pred = tf.squeeze(localization_pred,axis=0)
    
    score_pred = centerness_pred*classes_pred
#     score_pred = classes_pred
    result_list = []
    for thresh in range(0,40,2):
        mask = (score_pred > (thresh/100))

        _boxes = []
        _classes = []
        _scores = []
        for c in range(num_classes):
        ##nms
            _localization_pred = tf.boolean_mask(localization_pred,mask[:,c])
            _scores_pred = tf.boolean_mask(score_pred[:,c],mask[:,c])
            nms_index = tf.image.non_max_suppression(
                _localization_pred, _scores_pred, 10, iou_threshold = 0.5)
            _boxes.append(tf.gather(_localization_pred, nms_index))
            _scores_pred = tf.gather(_scores_pred,nms_index)
            _scores.append(_scores_pred)
            _classes.append(tf.ones_like(_scores_pred, 'int32') * c)
        _boxes = tf.concat(_boxes,axis=0)
        _scores = tf.concat(_scores,axis=0)
        _classes = tf.concat(_classes,axis=0)
        result_list.append([_scores,_classes,_boxes])
    return result_list
