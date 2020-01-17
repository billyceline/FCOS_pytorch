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
annotation_2012_train = get_path_and_annotation(train_2012_dir)
annotation_2007_train = get_path_and_annotation(train_2007_dir)
extra_annotation = get_path_and_annotation(val_2012_dir)
annotation_train = annotation_2012_train+annotation_2007_train+extra_annotation
random.shuffle(annotation_train)

# annotation_2012_test = get_path_and_annotation(FLAGS.test_2012_dir)
annotation_2007_test = get_path_and_annotation(test_2007_dir)
annotation_test = annotation_2007_test
print('The number of training samples are:'+str(len(annotation_train)))
train_iter = read_data(annotation_train, BATCH_SIZE, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), is_random=True, is_crop=True)
test_iter = read_data(annotation_test, 1, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), is_random=False, is_crop=False)
tsw = SummaryWriter(log_dir=LOG_DIR)

#####network
fcos = FCOS(FPN_Output, Head, backbone, num_classes=NUM_CLASSES).to(device)
###losses
fcos_loss = FCOSLoss(FocalLoss, IOULoss).to(device)
adam_optimizer = opt.Adam(fcos.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# sgd_optimizer = opt.SGD(fcos.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)


if(USE_CHECKPOINT):
    checkpoint = torch.load(CHECKPOINT_DIR+'resnet_18_iter_' + str(START_ITER) + '.pth.tar')
    print('Restore from the checkpoint!')
    fcos.load_state_dict(checkpoint['state_dict'])
else:
    START_ITER = 0

for iteration in range(START_ITER + 1, 1000000):
    if (((iteration - START_ITER) // BATCH_SIZE) % 16651 == 0):
        random.shuffle(annotation_train)
        train_iter = read_data(annotation_train, BATCH_SIZE, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH), is_random=True, is_crop=False)

    image_batch, annotation_batch, cls_batch = next(train_iter)
    #     image_batch = (image_batch-mean)/std
    # one hot for cls_batch(change class from[1,20] to [0,21])
    eye = np.eye(NUM_CLASSES)
    eye = np.concatenate([eye, np.expand_dims(np.zeros_like(eye[0]), axis=0)])
    cls_batch = eye[(cls_batch - 1).astype(np.int32)].astype(np.float32)
    matched_true_boxes, matched_true_classes, matched_true_centerness = encode_boxes(annotation_batch, cls_batch,
                                                                                     FEATURE_SIZE, STRIDE)
    feature_state = np.squeeze(matched_true_centerness != 0, axis=-1)

    # copy numpy ground truth data to GPU
    image_batch = torch.from_numpy(image_batch.astype(np.float32)).to(device).permute(0, 3, 1, 2)
    # torch.view(image_batch,[0,3,1,2])
    matched_true_boxes = torch.from_numpy(matched_true_boxes.astype(np.float32)).to(device)
    matched_true_classes = torch.from_numpy(matched_true_classes.astype(np.float32)).to(device)
    matched_true_centerness = torch.from_numpy(matched_true_centerness.astype(np.float32)).to(device)
    feature_state = torch.from_numpy(feature_state).to(device)
    # get network output for calculating losses
    centerness_output, classes_output, boxes_output = fcos.train()(image_batch, FEATURE_SIZE)
    # calculate loss
    total_losses, focal_losses, iou_losses, centerness_losses = fcos_loss(feature_state, matched_true_classes,
                                                                          matched_true_boxes, matched_true_centerness,
                                                                          classes_output, boxes_output,
                                                                          centerness_output)
    # clear gradient
    adam_optimizer.zero_grad()
    total_losses.backward()
    adam_optimizer.step()
    tsw.add_scalar('train/total_losses', total_losses, iteration)
    tsw.add_scalar('train/focal_losses', focal_losses, iteration)
    tsw.add_scalar('train/iou_losses', iou_losses, iteration)
    tsw.add_scalar('train/centerness_losses', centerness_losses, iteration)

    if (iteration % 50 == 0 and iteration != 0):
        print('Current step: ' + str(iteration) + '     Current total loss: ' + str(
            total_losses.cpu().data.numpy()) + '  focal loss: ' + str(
            focal_losses.cpu().data.numpy()) + ' iou loss: ' + str(
            iou_losses.cpu().data.numpy()) + ' centerness loss: ' + str(centerness_losses.cpu().data.numpy()))
    if (iteration % 200 == 0 and iteration != 0):
        torch.save({'state_dict': fcos.state_dict()},
                   CHECKPOINT_DIR+'resnet_18_iter_' + str(iteration) + '.pth.tar')
        print('Model saved!')

        #image_batch_test, annotation_batch_test, cls_batch_test = next(test_iter)
        #         image_batch_processed = (image_batch-mean)/std
        #image_batch_processed = torch.from_numpy(image_batch_test.astype(np.float32)).permute(0, 3, 1, 2).to(device)
        # centerness_output, classes_output, boxes_output = fcos.eval()(image_batch_processed, FEATURE_SIZE)
        #
        # temp_centerness_pred, temp_classes_pred, temp_localization_pred = predict_outputs(centerness_output,
        #                                                                                   classes_output, boxes_output,
        #                                                                                   FEATURE_SIZE, STRIDE,
        #                                                                                   inference_threshold=0.2)
        # xmin, ymin, xmax, ymax = np.split(temp_localization_pred.cpu().data.numpy(), axis=1, indices_or_sections=4)
        # plt.hlines(ymin, xmin, xmax, 'r')
        # plt.hlines(ymax, xmin, xmax, 'r')
        # plt.vlines(xmin, ymin, ymax, 'r')
        # plt.vlines(xmax, ymin, ymax, 'r')
        # for j in range(temp_centerness_pred.cpu().data.numpy().shape[0]):
        #     position = (ymin[j], xmin[j])
        #     plt.text(position[1] + 10, position[0] + 10, corresponding_dict[temp_classes_pred.cpu().data.numpy()[j]],
        #              color='g', size=10)
        # plt.imshow(image_batch_test[0])
        # plt.show()