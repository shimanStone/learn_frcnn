#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 14:44
# @Author  : shiman
# @File    : frcnn_training.py
# @describe:

import numpy as np
import torch.nn as nn


def weight_init(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m,'weight') and class_name.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
        elif class_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    print(f'initialize network with {init_type} type')

    net.apply(init_func)


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] !=4 or bbox_b.shape[1] !=4:
        print(bbox_a, bbox_b)
        raise ValueError
    tl = np.maximum(bbox_a[:,None,:2], bbox_b[:,:2])
    br = np.minimum(bbox_a[:,None,2:], bbox_b[:,2:])
    area_i = np.prod(br-tl, axis=1) * (tl< br).all(axis=2)
    area_a = np.prod(bbox_a[:,2:] - bbox_a[:,:2], axis=1)
    area_b = np.prod(bbox_b[:,2:] - bbox_a[:,:2], axis=1)

    return area_i / (area_a[:,None]+area_b - area_i)


class AnchorTargetCreator(object):
    def __init(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_iou, label = self._create_label(anchor, bbox)

    def _calc_ious(self, anchor, bbox):
        #
        ious = bbox_iou(anchor, bbox)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))

        # 获得每个先验框对应的真实框的iou值及索引位置
        max_ious = np.max(ious, axis=1)
        argmax_ious = ious.argmax(axis=1)
        # 获得每个真实框最对应的先验框
        gt_argmax_ious = ious.argmax(ious, axis=0)
        # 保证每个真实框都存在对应的先验框
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        # 1正样本，0负样本，-1忽略
        # 初始化都设为-1
        label = np.empty((len(anchor),),dtype=np.int32)
        label = label.fill(-1)
        #
        argmax_ious, max_ious, gt_argmax_iou = self._calc_ious(anchor,bbox)
        #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious > self.pos_iou_thresh] = 1
        if len(gt_argmax_iou) > 0:
            label[gt_argmax_iou] = 1

        # 判断正样本数量是否大于128，如果大于则限制128
        n_pos = int(self.pos_ratio*self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disabel_index = np.random.choice(pos_index, size=(len(pos_index)-n_pos),
                                             replace=False)
            label[disabel_index] = -1

        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label==1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disabel_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg),
                                             replace=False)
            label[disabel_index] = -1

        return argmax_ious, label


class ProposalTargetCreator():
    pass


class FasterRCNNTrainer(nn.Module):

    def __self__(self, model, optimizer):
        super().__init__()
        self.faster_rcnn = model
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalized_std = [0.1,0.1,0.2,0.2]

