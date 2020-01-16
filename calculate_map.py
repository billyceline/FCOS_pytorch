import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torchvision import models
from torchvision.ops import misc as misc_nn_ops
from util import *
from encode_boxes import *
from loss import *
from flags_and_variables import *
import torch.optim as opt
import os
from fcos import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from map import *

torch.cuda.empty_cache()
# annotation_2012_test = get_path_and_annotation(FLAGS.test_2012_dir)
annotation_2007_test = get_path_and_annotation(test_2007_dir)
annotation_test = annotation_2007_test
print('The number of testing samples are:'+str(len(annotation_test)))
test_iter = read_data(annotation_test,1,input_shape=(800,1024),is_random=False,is_crop=False)

#####network
fcos = FCOS(FPN_Output,Head,'resnet18',is_training=True,num_classes=20).to(device)
###losses

start_iter =26800
checkpoint = torch.load('/media/xinje/New Volume/fcos/pytorch/resnet_18/resnet_18_iter_'+str(start_iter)+'.pth.tar')
print('Restore from the checkpoint!')
fcos.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    gt_dict = {}
    pred_content = []
    for i in tqdm(range(len(annotation_test))):
        image_path = annotation_test[i].split(' ')[0]+' '+annotation_test[i].split(' ')[1]
        image_batch_test,annotation_batch_test,cls_batch_test = next(test_iter)
    #         image_batch_processed = (image_batch-mean)/std
        image_batch_processed = torch.from_numpy(image_batch_test.astype(np.float32)).permute(0,3,1,2).to(device)
        centerness_output,classes_output,boxes_output = fcos.eval()(image_batch_processed,feature_size)

        temp_score_pred,temp_classes_pred,temp_localization_pred = predict_outputs(centerness_output,classes_output,boxes_output,feature_size,stride,inference_threshold=0.15)
        temp_scores_pred_list = temp_score_pred.cpu().data.numpy()
        temp_classes_pred_list = temp_classes_pred.cpu().data.numpy()
        temp_localization_pred_list = temp_localization_pred.cpu().data.numpy()

        mask = (cls_batch_test-1)>=0
        cls_batch_test = (cls_batch_test[mask]-1).astype(np.int32)
        annotation_batch_test = annotation_batch_test[mask]*np.array([image_height,image_width,image_height,image_width])
        gt_dict[image_path] = np.concatenate([annotation_batch_test,np.expand_dims(cls_batch_test,axis=-1)],axis=-1)

        for k in range(temp_classes_pred_list.shape[0]):
            temp_x_min,temp_y_min,temp_x_max,temp_y_max=temp_localization_pred_list[k]
            score = temp_scores_pred_list[k]
            label = temp_classes_pred_list[k]
            pred_content.append([image_path, temp_y_min, temp_x_min, temp_y_max, temp_x_max, score, label])

ap_list = []
recall_list = []
precision_list =[]
for i in range(num_classes):
    ap,recall,precision = voc_eval(gt_dict, pred_content, i, iou_thres=0.5, use_07_metric=False)
    ap_list.append(ap)
    recall_list.append(recall)
    precision_list.append(precision)
print('MAP:')
print(np.array(ap_list).mean())
print('AP:')
class_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car',\
                          'chair','cow','dog','horse','motorbike',\
                         'pottedplant','sheep','sofa','train','tvmonitor','cat','person','diningtable']
AP = dict(zip(class_list,ap_list))
print(AP)