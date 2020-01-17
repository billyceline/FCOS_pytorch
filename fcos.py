import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN,resnet_fpn_backbone
from flags_and_variables import *

class Head(nn.Module):
    def __init__(self,name,num_classes):
        super(Head,self).__init__()
        self.block_1 = torch.nn.Sequential()
        for j in range(4):
            self.block_1.add_module(name+"_block"+str(1)+"_conv"+str(j),torch.nn.Conv2d(256,256,3,stride=1,padding=[1,1]))
            self.block_1.add_module(name+"_block"+str(1)+"_bn"+str(j),torch.nn.BatchNorm2d(256,affine=True))
            self.block_1.add_module(name+"_block"+str(1)+"_relu"+str(j),torch.nn.ReLU())
            
        self.block_2 = torch.nn.Sequential()
        for j in range(4):
            self.block_2.add_module(name+"_block"+str(2)+"_conv"+str(j),torch.nn.Conv2d(256,256,3,stride=1,padding=[1,1]))
            self.block_2.add_module(name+"_block"+str(2)+"_bn"+str(j),torch.nn.BatchNorm2d(256,affine=True))
            self.block_2.add_module(name+"_block"+str(2)+"_relu"+str(j),torch.nn.ReLU())        
        
        self.centerness_ouput = torch.nn.Conv2d(256, 1, 3, stride=1, padding=[1, 1])
        self.classes_ouput = torch.nn.Conv2d(256, num_classes, 3, stride=1, padding=[1, 1])
        torch.nn.init.constant_(self.classes_ouput.bias, -np.log((1-0.01)/0.01))
        self.localization_ouput = torch.nn.Conv2d(256, 4, 3, stride=1, padding=[1, 1])

    def forward(self, fpn_output, feature_size):
        layer1 = self.block_1(fpn_output)
        centerness_tensor = self.centerness_ouput(layer1)
        classes_tensor = self.classes_ouput(layer1)
###
        centerness_tensor = centerness_tensor.permute(0, 2, 3, 1)
        classes_tensor = classes_tensor.permute(0, 2, 3, 1)
        centerness_tensor = torch.reshape(centerness_tensor, (-1, (feature_size[0]*feature_size[1]), 1))
        classes_tensor = torch.reshape(classes_tensor, (-1, (feature_size[0]*feature_size[1]), NUM_CLASSES))
###
        #centerness_tensor = torch.reshape(centerness_tensor,(-1,1,(feature_size[0]*feature_size[1])))
        #classes_tensor = torch.reshape(classes_tensor,(-1,num_classes,(feature_size[0]*feature_size[1])))

        layer2 =  self.block_2(fpn_output)
        localization_tensor = self.localization_ouput(layer2)
####
        localization_tensor = localization_tensor.permute(0, 2, 3, 1)
####
        localization_tensor = torch.reshape(localization_tensor, (-1, (feature_size[0]*feature_size[1]), 4))
        return centerness_tensor, classes_tensor, localization_tensor


class FPN_Output(nn.Module):
    def __init__(self, fpn, backbone_name):
        super(FPN_Output, self).__init__()
        self.fpn = resnet_fpn_backbone(backbone_name=backbone_name,pretrained=True)
        self.conv2d_feature4 = torch.nn.Conv2d(256,256,3,stride=2,padding=[1,1])
        self.conv2d_feature5 = torch.nn.Conv2d(256,256,3,stride=2,padding=[1,1])

    def forward(self,image_batch):
        fpn_output = self.fpn(image_batch)
        fpn_output[4] = self.conv2d_feature4(fpn_output[3])
        fpn_output[5] = self.conv2d_feature5(fpn_output[4])
        return fpn_output


class FCOS(nn.Module):
    def __init__(self,FPN_Output,Head,backbone_name, num_classes=NUM_CLASSES):
        super(FCOS,self).__init__()
        self.fpn_output = FPN_Output(resnet_fpn_backbone, backbone_name)
        self.head1 = Head('head1',num_classes = num_classes)
        self.head2 = Head('head2',num_classes = num_classes)
        self.head3 = Head('head3',num_classes = num_classes)
        self.head4 = Head('head4',num_classes = num_classes)
        self.head5 = Head('head5',num_classes = num_classes)

    def forward(self, image_batch, feature_size):
        fpn_outputs = self.fpn_output(image_batch)
        centerness_tensor1, classes_tensor1, localization_tensor1 = self.head1(fpn_outputs[1], feature_size[0])#(N,100,128)
        centerness_tensor2, classes_tensor2, localization_tensor2 = self.head2(fpn_outputs[2], feature_size[1])#(N,50,64)
        centerness_tensor3, classes_tensor3, localization_tensor3 = self.head3(fpn_outputs[3], feature_size[2])#(N,25,32)
        centerness_tensor4, classes_tensor4, localization_tensor4 = self.head4(fpn_outputs[4], feature_size[3])#(N,13,16)
        centerness_tensor5, classes_tensor5, localization_tensor5 = self.head5(fpn_outputs[5], feature_size[4])#(N,7,8)
        centerness_output = [centerness_tensor1, centerness_tensor2, centerness_tensor3, centerness_tensor4, centerness_tensor5]
        class_output = [classes_tensor1, classes_tensor2, classes_tensor3, classes_tensor4, classes_tensor5]
        localization_output = [localization_tensor1, localization_tensor2, localization_tensor3, localization_tensor4, localization_tensor5]
        
        centerness = torch.cat(centerness_output,1)
        classes = torch.cat(class_output,1)
        localization = torch.cat(localization_output,1)

        centerness = torch.sigmoid(centerness)
        classes = torch.sigmoid(classes)
        localization = torch.exp(localization)
        
        return centerness, classes, localization

