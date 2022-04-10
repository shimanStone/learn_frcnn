#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 14:31
# @Author  : shiman
# @File    : classifier.py
# @describe:

import torch
import torch.nn as nn
from torchvision.ops import RoIPool

import warnings
warnings.filterwarnings("ignore")


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class Resnet50RoIHead(nn.Module):
    """进行ROIPooling + layer4 + avepool, 进行预测和回归分析 """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super().__init__()

        self.classifier = classifier
        # 对 roi pooling后结果进行回归预测
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # 对 roi pooling后结果进行分类预测
        self.score = nn.Linear(2048, n_class)
        # 权重初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, rois_indices, img_size):
        n, _, _, _ = x.shape  # 公用特征层
        if x.is_cuda:
            rois_indices = rois_indices.cuda()
            rois = rois.cuda()
        # 获取公用特征层上的建议框
        rois_feature_map = torch.zeros_like(rois)  # shanpe(num_rois, 4)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]
        #
        indices_and_rois = torch.cat([rois_indices[:, None], rois_feature_map], dim=1)
        # 利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois)  # shape(num_rois, roi_size, roi_size, 1024)
        # 当输入为一张图片时 (layer7 stride 2 + avegpool kernel size 7) shape:n_rois, 2048,1,1
        fc7 = self.classifier(pool)
        # shape 300, 2048  (降维)
        fc7 = fc7.view(fc7.size(0), -1)
        #
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        #
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))  # 回归预测，对建议框进行调整
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))  # 分类预测，对建议框进行分类判断

        return roi_cls_locs, roi_scores