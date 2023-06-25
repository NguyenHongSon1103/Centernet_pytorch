# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import _transpose_and_gather_feat, Decoder, _topk, _nms
import math

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    pred = torch.clip(pred, 1e-4, 1.-1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
        Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask
        
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''
    def __init__(self):
        super(RegLoss, self).__init__()
    
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
    
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

# IOU - base loss

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class BboxLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, size, output, mask, ind, target):
        """IoU loss."""
        # Get pred_bboxes
        pred_wh = _transpose_and_gather_feat(output, ind)
        x, y = (ind % size).unsqueeze(2), (ind // size).unsqueeze(2)
        pred_boxes = torch.cat([x, y, pred_wh], -1)
        target_boxes = torch.cat([x, y, target], -1)
        iou = bbox_iou(pred_boxes[mask], target_boxes[mask], xywh=True, CIoU=True)
        loss_iou = ((1.0 - iou)).sum() / (mask.sum() + 1e-4)

        return loss_iou
    
class Loss(nn.Module):
    def __init__(self, weights=[1.0, 0.1, 1.0]) -> None:
        super().__init__()
        self.hm_loss = FocalLoss()
        # self.iou_loss = BboxLoss()
        self.wh_loss = RegLoss()
        self.reg_loss = RegLoss()
        self.loss_keys = ['hm_loss', 'wh_loss', 'reg_loss', 'total_loss']
        self.weights = weights
    
    def forward(self, output, target):
        '''
        output: 
                hm: NxCxWxH | wh: Nx2xWxH | reg: Nx2xWxH
        target: 
                hm: NxCxWxH | wh: NxKx2 | reg: NxKx2 | indices: NxK
        '''
        # for ts in target:
        #     print(ts.shape)
        
        with torch.no_grad():
            indices = target[3].type(torch.int64)
            mask = target[3] > 0
        hm_loss = self.hm_loss(output[0], target[0].permute(0, 3, 1, 2)) * self.weights[0]
        wh_loss = self.wh_loss(output[1], mask, indices, target[1])*self.weights[1]
        # iou_loss = self.iou_loss(output[1].shape[2], output[1], mask, indices, target[1]) #for ciou loss
        # wh_loss = wh_loss + iou_loss
        
        reg_loss = self.reg_loss(output[2], mask, indices, target[2])*self.weights[2]

        total_loss = hm_loss + wh_loss + reg_loss
        loss_dict = {key:value for key, value in zip(self.loss_keys, [hm_loss, wh_loss, reg_loss, total_loss])}
       
        return total_loss, loss_dict
    
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, output, target, mask = None):
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, output, target):
        loss = F.mse_loss(output, target, reduction='mean')
        # loss = loss / (target.sum() + 1e-4)
        return loss
    
class UnsupLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.hm_loss = FocalLoss()
        self.hm_loss = MSELoss()
        self.wh_loss = L1Loss()
        self.reg_loss = L1Loss()
        self.loss_keys = ['unsup_hm_loss', 'unsup_wh_loss', 'unsup_reg_loss', 'unsup_total_loss']
    
    def forward(self, output_s, output_t, current_epoch):
        '''
        output: 
                hm_s: NxCxWxH | wh_s: Nx2xWxH | reg_s: Nx2xWxH
        target: 
                hm_t: NxCxWxH | wh_t: Nx2xWxH | reg_t: Nx2xWxH
        '''
        ## Get topk index
        # with torch.no_grad():
        #     heat = _nms(output_t[0])
        #     #Get 10 peak, slowly increase threshold
        #     sorted_vals, sorted_inds = torch.topk(heat.view(-1), 10)
        #     topk_val = max(0.05 + (0.2-0.05)*current_epoch/100, float(sorted_vals[-1]))
        #     heat = torch.where(heat >  topk_val, torch.ones_like(heat), heat)
        
        with torch.no_grad():
            # heat = _nms(output_t[0]) # NxCxHxW
            heat = output_t[0]
            for i in range(heat.shape[0]):
                #Get 5 peak, slowly increase threshold
                sorted_vals, sorted_inds = torch.topk(heat[i].view(-1), 5)
                # print(sorted_vals)
                # Keep atleast 1 peak to pseudo target
                # topk_val = min(sorted_vals[0], max(0.2 + 0.8*current_epoch/100, float(sorted_vals[-1])))
                # heat[i] = torch.where(heat[i] >=  topk_val, torch.ones_like(heat[i]), heat[i])
                
                #Thử nghiệm với ngưỡng cố định: 0.1, 0.2, 0.3, 0.4, 0.5
                heat[i] = torch.where(heat[i] >=  0.4, torch.ones_like(heat[i]), heat[i])
                
            mask = heat.max(1, keepdim=True)[0]
            
        hm_loss = self.hm_loss(output_s[0], heat) 
        wh_loss = self.wh_loss(output_s[1], output_t[1], mask)*0.1 
        reg_loss = self.reg_loss(output_s[2], output_t[2], mask)
        
        total_loss = hm_loss + wh_loss + reg_loss
        loss_dict = {key:value for key, value in zip(self.loss_keys, [hm_loss, wh_loss, reg_loss, total_loss])}
        
        # total_loss = hm_loss
        # loss_dict = {'unsup_hm_loss':hm_loss, 'unsup_total_loss':total_loss}
       
        return total_loss, loss_dict
