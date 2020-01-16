import torch
import torch.nn as nn
from flags_and_variables import *

class FocalLoss(nn.Module):
    def __init__(self, loss_type="focal"):
        super(FocalLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, y_true, y_pred,location_state, weight=None):
        alpha=0.25
        gamma=2.0
        true_label = torch.ones_like(y_true)

        alpha_factor = torch.ones_like(y_true) * alpha
        alpha_factor = torch.where(torch.eq(y_true, true_label), alpha_factor, true_label - alpha_factor)

        focal_weight = torch.where(torch.eq(y_true, true_label), true_label - y_pred, y_pred)
        focal_weight = -alpha_factor * (focal_weight ** gamma)
        cls_loss = torch.where(torch.eq(y_true, true_label),focal_weight * y_pred.log(),focal_weight * ((1-y_pred).log()))
        #cls_loss = torch.mean(cls_loss,-1)
        #criterion = nn.BCELoss(reduction='none')
        #cls_loss = (focal_weight * criterion(y_pred,y_true))

        # compute the normalizer: the number of positive anchors
        # normalizer = torch.eq(location_state, torch.ones_like(location_state))
        # normalizer = normalizer[0].sum()
        normalizer = torch.max(torch.tensor(1).to(device), location_state.sum())
        loss = cls_loss.sum() / normalizer
        return loss

class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target,feature_state, weight=None):
        pred_top = pred[..., 0]
        pred_bottom = pred[..., 1]
        pred_left = pred[..., 2]
        pred_right = pred[..., 3]

        target_top = target[..., 0]
        target_bottom = target[..., 1]
        target_left = target[..., 2]
        target_right = target[..., 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect*(800*1024) + 1.0) / (area_union*(800*1024) + 1.0)
        gious = ious - (ac_uion - area_union) / (ac_uion)
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

class FCOSLoss(nn.Module):
    def __init__(self,FocalLoss,IOULoss,loss_type="fcos_loss"):
        super(FCOSLoss, self).__init__()
        self.loss_type = loss_type
        self.focalloss = FocalLoss(loss_type='focal')
        self.iouloss = IOULoss(loss_type="iou")
        self.BCELoss=nn.BCELoss(reduction='none')

    def forward(self,feature_state,matched_true_classes,matched_true_boxes,matched_true_centerness,classes,localization,centerness):
        index = torch.where(feature_state)
        focal_losses = self.focalloss(matched_true_classes, classes,feature_state)
        iou_losses = self.iouloss(localization[index],matched_true_boxes[index],feature_state)/feature_state.sum()
        centerness_losses = (self.BCELoss(centerness[index],matched_true_centerness[index])).sum()/feature_state.sum()
        total_loss = focal_losses+iou_losses+centerness_losses
        return total_loss,focal_losses,iou_losses,centerness_losses
