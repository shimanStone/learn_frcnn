#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 14:03
# @Author  : shiman
# @File    : frcnn_net.py
# @describe:

import torch.nn as nn
from .rpn import RegionProposalsNetwork
from .resnet import resnet50
from .classifier import Resnet50RoIHead, VGG16RoIHead
from .vgg16 import decom_vgg16


class FasterRCNN(nn.Module):

    def __init__(self, num_classes, mode='training', feat_stride=16, anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2], backbone='vgg16', pretrained=False):
        super().__init__()

        self.feat_stride = feat_stride
        if backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            # 构建classifier
            self.rpn = RegionProposalsNetwork(in_channels=1024, mid_channels=512,
                                              ratios=ratios, anchor_scales=anchor_scales,
                                              feat_stride=self.feat_stride, mode=mode)
            self.head = Resnet50RoIHead(n_class=num_classes + 1, roi_size=14,
                                        spatial_scale=1, classifier=classifier)
        if backbone == 'vgg16':
            self.extractor, classifier = decom_vgg16(pretrained)
            self.rpn = RegionProposalsNetwork(in_channels=512, mid_channels=512,
                                              ratios=ratios, anchor_scales=anchor_scales,
                                              feat_stride = self.feat_stride, mode=mode)
            self.head = VGG16RoIHead(n_class=num_classes+1, roi_size=7,
                                     spatial_scale=1, classifier=classifier)


    def forward(self, x, scale=1.):
        # 输入图片的大小
        img_size = x.shape[2:]
        # 主干网络提取特征
        base_feature = self.extractor.forward(x)
        # 获取建议框
        _, _, rois, rois_indices, _ = self.rpn.forward(base_feature, img_size, scale)
        # 获得classifier的分类结果和回归结果
        roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, rois_indices, img_size)

        return roi_cls_locs, roi_scores, rois, rois_indices

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()