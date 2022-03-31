#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 13:55
# @Author  : shiman
# @File    : utils_bbox.py
# @describe:

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms



def loc2bbox(src_bbox, loc):
    """将先验框网格调整成建议框"""
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)
    # src_bbox shape (38*38*9, 4)
    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], dim=-1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], dim=-1)
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], dim=-1) + 0.5 * src_width  # shape(38*38*9, 1)
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], dim=-1) + 0.5 * src_height

    # loc shape (38*38*9,4)
    dx = loc[:, 0::4]  # shape(38,38*9,1)
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    # shape (38*38*9, 1)
    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(src_bbox)

    dst_bbox[:, 0::4] = ctr_x - w * 0.5
    dst_bbox[:, 2::4] = ctr_x + w * 0.5
    dst_bbox[:, 1::4] = ctr_y - h * 0.5
    dst_bbox[:, 3::4] = ctr_y + h * 0.5

    return dst_bbox


class DecodeBox(object):
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1

    def frcnn_correct_boxes(self, normal_box, image_shape):
        # normal(left,top,right,botton) -> image(top,left,bottom,right)
        h, w = image_shape
        image_box = np.ones_like(normal_box, dtype='int32')
        image_box[:, 0], image_box[:, 2] = normal_box[:, 1] * h, normal_box[:, 3] * h
        image_box[:, 1], image_box[:, 3] = normal_box[:, 0] * w, normal_box[:, 2] * w

        return image_box

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                nms_iou=0.3, confidence=0.5):

        results = []
        bs = len(roi_cls_locs)
        rois = rois.view((bs, -1, 4))  # shape: batch size, num_rois, 4
        # 对每一张图片进行处理
        for i in range(bs):
            # 对回归参数进行reshape
            roi_cls_loc = roi_cls_locs[i] * self.std
            # shape 框个数，每个种类，对应种类的调整参数
            roi_cls_loc = roi_cls_loc.view((-1, self.num_classes, 4))
            # 对建议框进行调整得到预测框
            # rois (num_rois,4) -> (num_rois,1,4) -> (num_rois, num_class, 4)
            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox = cls_bbox.view((-1, self.num_classes, 4))
            # 对预测框进行归一化调整，至0-1
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                """取出属于该类的所有框的置信度，判断是否大于阈值"""
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence 的框
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(boxes_to_process, confs_to_process, nms_iou)
                    # 取出在非极大值抑制中效果最好的内容
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:,None]
                    labels = (c - 1) * torch.ones((len(keep), 1))
                    labels = labels.cuda() if confs.is_cuda else labels
                    # 将label、conf、box堆叠
                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()

                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])  # 归一化的(-w,-h,w,h)
                # 获得在原始图像上的位置
                results[-1][:, :4] = self.frcnn_correct_boxes(results[-1][:, :4], image_shape)

        return results