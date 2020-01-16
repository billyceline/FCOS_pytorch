import torch
import numpy as np

image_height = 800
image_width = 1024
batch_size = 14
weight_decay = 1e-5
learning_rate= 5e-7
log_Dir = './log'
backbone = 'resnet18'
#checkpoint_dir = '/media/xinje/New Volume/fcos/resnet_v2_50_freezed_backbone/'
feature_size=[(100,128),(50,64),(25,32),(13,16),(7,8)]
feature_layer_list = [1,2,3,4,5]#P3,P4,P5,P6,P7
stride=[8,16,32,64,128]
is_training= True
num_classes=20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_2012_dir = '/media/xinje/New Volume/VOC07&12/VOC2012/train_2012.txt'
train_2007_dir = '/media/xinje/New Volume/VOC07&12/VOC2007/train/train_2007.txt'
val_2012_dir = '/media/xinje/New Volume/VOC07&12/VOC2012/val_2012.txt'
test_2007_dir = '/media/xinje/New Volume/VOC07&12/VOC2007/test/test_2007.txt'
inference_threshold = 0.2

corresponding_dict = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',\
                      7:'chair',8:'cow',19:'diningtable',9:'dog',10:'horse',11:'motorbike',\
                     12:'pottedplant',13:'sheep',14:'sofa',15:'train',16:'tvmonitor',17:'cat',18:'person'}
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
