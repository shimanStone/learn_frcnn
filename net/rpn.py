#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 9:44
# @Author  : shiman
# @File    : rpn.py
# @describe:

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision.ops import nms

from fastercnn.utils.utils_bbox import loc2bbox



def generate_anchor_base(base_size=16, anchor_scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    """生成基础先验框"""
    anchor_base = np.zeros((len(anchor_scales) * len(ratios), 4), dtype=np.float32)
    index = 0
    for i in ratios:
        for j in anchor_scales:
            h = base_size * j * np.sqrt(i)
            w = base_size * j * np.sqrt(1. / i)

            anchor_base[index, 0] = -h / 2.
            anchor_base[index, 1] = -w / 2
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
            index += 1

    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """对基础先验框进行拓展对应到所有特征点上 ，框对应在影像上的位置
       return: (anchor_num, 4)
       - anchor_num = height*width*9
       - 4:(-w, -h, w, h) 左上和右下点坐标
       """

    # 计算每个网格（对应在原始图片上）的中心点（网格特征点对应到原始图像上的位置）
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()),
                     axis=1)  # shift.shape:(width*height, 4)
    # 计算每个网格上的九个先验框
    A = anchor_base.shape[0]  # 每个网格的先验框个数
    K = shift.shape[0]  # 网格个数

    anchor_r = anchor_base.reshape((1, A, 4))  # 每个网格，先验框个数，框范围
    shift_r = shift.reshape((K, 1, 4))  # 总的网格数， 每个先验框， 每个网格数的中心点

    anchor = anchor_r + shift_r  # （K,A,4） # 总的网格个数，先验框个数，纠正后的框范围
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class ProposalCreator():
    def __init__(self, mode, nms_iou=0.7, n_train_pre_nms=12000, n_train_post_nms=600,
                 n_test_pre_nms=3000, n_test_post_nms=300, min_size=16):
        """
        mode: 训练还是预测
        nms_iou：非极大值抑制重叠区域阈值
        n_pre_nms：所有框得分排序后取前多少个框的个数, 取出的建议框
        n_post_nms：对建议框进行非极大值抑制， 后再根据排序取出多少个框
        min_size: 建议框最小宽高数
        """

        # 设置预测还是训练
        self.mode = mode
        # 建议框非极大值抑制 iou大小
        self.nms_iou = nms_iou
        # 训练用到的建议框数量
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        # 预测用到的建议框数量
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """
        先验框 -> (loc) -> 建议框1 -> (范围及宽高) -> 建议框2 -> (score+n_pro_nms) -> 建议框3
        -> (score+nms) -> 建议框4 -> (score+n_post_nms) -> 建议框5
        """
        if self.mode == 'training':
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        anchor = torch.from_numpy(anchor)  # tensor(38*38*9,4)
        if loc.is_cuda:
            anchor = anchor.cuda()

        # 将rpn先验框网络结果转换成建议框
        roi = loc2bbox(anchor, loc)
        # 对建议框进行处理
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # 建议框的宽高最小值不可以小于16
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]

        roi = roi[keep, :]
        score = score[keep]

        # 进行得分排序，取出建议框
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]

        roi = roi[order, :]
        score = score[order]

        # 对建议框进行非极大抑制
        # 1、选择现有框中得分最高的框，遍历其他的框，如果和当前最高得分的框的覆盖范围大于阈值，则删除
        # 2、从未选择的框中选择最高得分的框再重复上述工作，直到所有框都完成
        keep = nms(roi, score, self.nms_iou)

        keep = keep[:n_post_nms]
        roi = roi[keep, :]

        return roi


class RegionProposalsNetwork(nn.Module):

    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 feat_stride=16, mode='training'):
        super().__init__()
        # 基础先验框（9*4）
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]
        # 3*3，卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 1*1，n_anchor*2卷积, 分类预测
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 1*1 n_anchor*4卷积， 回归预测
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # 特征点间距步长
        self.feat_stride = feat_stride  # 特征步长，原始图像到共享特征层进行了四次压缩（2**4）
        # 对建议框进行非极大抑制
        self.proposal_layer = ProposalCreator(mode)
        # 对FPN网络部分进行权重初始化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # x: feature map
        n, c, h, w = x.shape
        # 3*3卷积 特征提取
        x = F.relu(self.conv1(x))
        # 回归预测
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # (batch_size, 先验框， 4)
        # 分类预测
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # softmax 概率计算
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()  # 包含物体的概率（置信度）
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        # 生成先验框，此时获取的anchor是布满所有网格点的，当输入图片（600，600，3） -> (38,38,) 1444个网格点-> 1444*9 ->（12996，4）
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        rois, rois_indices = list(), list()
        for i in range(n):
            # 先验框 -> 建议框
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)  # (n_post_nms,4)
            batch_index = i * torch.ones((len(roi),))  # [i * 1 for _ in range(n_post_nms)]

            rois.append(roi)
            rois_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)

        rois_indices = torch.cat(rois_indices, dim=0)

        return rpn_locs, rpn_scores, rois, rois_indices, anchor