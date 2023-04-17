# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 10:17 下午
# @Author  : Haonan Wang
# @File    : Losses.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        truth=truth.to(torch.float32)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.2, 0.8]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        logit = logit.squeeze(0)
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=0.5):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight
    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        inputs = inputs.max(1)[0]
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal

        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE
        # print "dice_focal",dice_focal_loss
        return dice_BCE_loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_label=255):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, inputs, targets):
        loss = self.loss(F.log_softmax(inputs, 1), targets)
        return loss
        # return 1.0*loss1 + 0.4*loss2 + 0.4*loss3 + 0.4*loss4
    
class LBTW_Loss(nn.Module):

    def __init__(self, Loss):
        super(LBTW_Loss, self).__init__()
        self.criteria = Loss
        try:
            self.__name__ = "LBTW_" + self.criteria.__name__
        except:
            self.__name__ = "LBTW_" + self.criteria._get_name()
    def _get_name(self):
        super(LBTW_Loss, self)._get_name()
        return "LBTW_" + self.criteria._get_name()


    def forward(self, pred1, pred2, pred3, pred4, target):
        outs=[pred1, pred2, pred3, pred4]
        sub_loss = []
        for i in range(4):
            pred=outs[i]
            curr_loss = self.criteria(pred, target)
            sub_loss.append(curr_loss)
        # print sub_loss[0].data, sub_loss[1].data, sub_loss[2].data, sub_loss[3].data
        return sub_loss, sub_loss[0], sub_loss[1], sub_loss[2], sub_loss[3]

class LBTW_algorithm():
    def __init__(self):
        self.initial_task1_loss_list = 0
        self.initial_task2_loss_list = 0
        self.initial_task3_loss_list = 0
        self.initial_task4_loss_list = 0
        self.weights_out1_save = []
        self.weights_out2_save = []
        self.weights_out3_save = []
        self.weights_out4_save = []

    def __call__(self, batch_num,loss1, loss2, loss3, loss4, alpha=0.5):
        if batch_num == 0:
            self.initial_task1_loss_list = loss1.item()
            self.initial_task2_loss_list = loss2.item()
            self.initial_task3_loss_list = loss3.item()
            self.initial_task4_loss_list = loss4.item()
        out1_loss_ratio = loss1.item() / self.initial_task1_loss_list
        out2_loss_ratio = loss2.item() / self.initial_task2_loss_list
        out3_loss_ratio = loss3.item() / self.initial_task3_loss_list
        out4_loss_ratio = loss4.item() / self.initial_task4_loss_list

        out1_loss_weight = pow(out1_loss_ratio,alpha)
        out2_loss_weight = pow(out2_loss_ratio,alpha)
        out3_loss_weight = pow(out3_loss_ratio,alpha)
        out4_loss_weight = pow(out4_loss_ratio,alpha)

        weights_sum = out1_loss_weight+out2_loss_weight+out3_loss_weight+out4_loss_weight
        out1_loss_weight  = out1_loss_weight / weights_sum * 4
        out2_loss_weight = out2_loss_weight / weights_sum * 4
        out3_loss_weight = out3_loss_weight / weights_sum * 4
        out4_loss_weight = out4_loss_weight / weights_sum * 4

        self.weights_out1_save.append(out1_loss_weight)
        self.weights_out2_save.append(out2_loss_weight)
        self.weights_out3_save.append(out3_loss_weight)
        self.weights_out4_save.append(out4_loss_weight)

        w1 = self.weights_out1_save
        w2 = self.weights_out2_save
        w3 = self.weights_out3_save
        w4 = self.weights_out4_save
        losses = loss1 * out1_loss_weight \
                 + loss2 * out2_loss_weight \
                 + loss3 * out3_loss_weight \
                 + loss4 * out4_loss_weight
        return losses / 4.0, w1, w2, w3, w4


