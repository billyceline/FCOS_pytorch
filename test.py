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
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()
# annotation_2012_test = get_path_and_annotation(FLAGS.test_2012_dir)
annotation_2007_test = get_path_and_annotation(test_2007_dir)
annotation_test = annotation_2007_test
print('The number of testing samples are:'+str(len(annotation_test)))
test_iter = read_data(annotation_test, 1, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), is_random=False, is_crop=False)

#####network
fcos = FCOS(FPN_Output, Head, backbone, num_classes=NUM_CLASSES).to(device)
###losses

checkpoint = torch.load('/media/xinje/New Volume/fcos/pytorch/resnet_18/resnet_18_iter_'+str(START_ITER)+'.pth.tar')
print('Restore from the checkpoint!')
fcos.load_state_dict(checkpoint['state_dict'])

for i in range(100):
    image_batch_test, annotation_batch_test, cls_batch_test = next(test_iter)
            # image_batch_processed = (image_batch-mean)/std
    image_batch_processed = torch.from_numpy(image_batch_test.astype(np.float32)).permute(0, 3, 1, 2).to(device)
    centerness_output, classes_output, boxes_output = fcos.eval()(image_batch_processed, FEATURE_SIZE)

    temp_centerness_pred, temp_classes_pred, temp_localization_pred = predict_outputs(centerness_output,
                                                                                      classes_output, boxes_output,
                                                                                      FEATURE_SIZE, STRIDE,
                                                                                      inference_threshold=0.2)
    xmin, ymin, xmax, ymax = np.split(temp_localization_pred.cpu().data.numpy(), axis=1, indices_or_sections=4)
    plt.hlines(ymin, xmin, xmax, 'r')
    plt.hlines(ymax, xmin, xmax, 'r')
    plt.vlines(xmin, ymin, ymax, 'r')
    plt.vlines(xmax, ymin, ymax, 'r')
    for j in range(temp_centerness_pred.cpu().data.numpy().shape[0]):
        position = (ymin[j], xmin[j])
        plt.text(position[1] + 10, position[0] + 10, corresponding_dict[temp_classes_pred.cpu().data.numpy()[j]],
                 color='g', size=10)
    plt.imshow(image_batch_test[0])
    plt.show()