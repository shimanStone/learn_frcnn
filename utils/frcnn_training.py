#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 14:44
# @Author  : shiman
# @File    : frcnn_training.py
# @describe:

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


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
    area_i = np.prod(br-tl, axis=2) * (tl< br).all(axis=2)
    area_a = np.prod(bbox_a[:,2:] - bbox_a[:,:2], axis=1)
    area_b = np.prod(bbox_b[:,2:] - bbox_b[:,:2], axis=1)

    return area_i / (area_a[:,None]+area_b - area_i)

def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


class AnchorTargetCreator(object):
    """
    input: anchor(?????????,shape:(n,4))??? bbox(?????????(m,4))
        ious: anchor???bbox????????????,shape(n,m)
        labels: pos\neg???????????????n_sample??????pos < n_sample*pos_ratio (1????????????0????????????-1??????)
    output:  ??????????????????????????????????????????(??????)loc, label
    """
    # ???????????????????????????????????????????????????????????????????????????

    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor):
        argmax_ious, label = self._create_label(anchor, bbox)  #
        if (label > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        # ?????????????????????????????????????????????
        ious = bbox_iou(anchor, bbox) # shape:(num_anchor,num_bbox)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))

        # ??????????????????????????????????????????iou??????????????????
        max_ious = np.max(ious, axis=1)
        argmax_ious = ious.argmax(axis=1)
        # ??????????????????????????????????????????
        gt_argmax_ious = ious.argmax(axis=0)
        # ????????????????????????????????????????????????
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        # 1????????????0????????????-1??????
        # ??????????????????-1
        label = np.empty((len(anchor),),dtype=np.int32)
        label.fill(-1)
        #
        argmax_ious, max_ious, gt_argmax_iou = self._calc_ious(anchor,bbox)
        #
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_iou) > 0:
            label[gt_argmax_iou] = 1

        # ?????????????????????????????????128????????????????????????128
        n_pos = int(self.pos_ratio*self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disabel_index = np.random.choice(pos_index, size=(len(pos_index)-n_pos),
                                             replace=False)
            label[disabel_index] = -1

        # ???????????????????????????????????????256
        n_neg = self.n_sample - np.sum(label==1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disabel_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg),
                                             replace=False)
            label[disabel_index] = -1

        return argmax_ious, label


class ProposalTargetCreator():
    """
    ???600???rois??????ground truth (???????????????128???)
    input: rois(600????????????)????????????????????????box(R,4), ??????box????????????label(R,1)
    output: 128???sample rois, ????????????gt_roi_loc, gt_roi_label
    """
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5,
                 neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample*self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1,0.1,0.2,0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ???????????????????????????????????????
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            gt_assignment = iou.argmax(axis=1)  # ???????????????????????????index
            max_iou = iou.max(axis=1)  # ??????????????????????????????iou
            gt_roi_label = label[gt_assignment] + 1  # ???????????????+1???????????????????????????

        # ?????????????????????????????????????????????neg_iou_thresh_high??????????????????
        # ??????????????????????????????self.pos_roi_per_image??????
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        # ??????????????????????????????????????????neg_hight,??????neg_low??????????????????
        neg_index = np.where((max_iou < self.neg_iou_thresh_high)&(max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        #
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0

        return sample_roi, gt_roi_loc, gt_roi_label


class FasterRCNNTrainer(nn.Module):

    def __init__(self, model, optimizer, loss_history):
        super().__init__()
        self.faster_rcnn = model
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalized_std = [0.1,0.1,0.2,0.2]

        self.log_info = loss_history

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]

        sigma_squared = sigma **2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = regression_diff.abs()
        regression_loss = torch.where(
            regression_diff < (1./sigma_squared),
            0.5*sigma_squared*regression_diff**2,
            regression_diff - 0.5/sigma_squared
        )
        regression_loss = regression_loss.sum()
        num_pos = (gt_label>0).sum().float()

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss

    def forward(self, imgs, bboxes, labels, scale):
        n, img_size = imgs.shape[0], imgs.shape[2:]
        # ???????????????
        base_feature = self.faster_rcnn.extractor(imgs)
        # ??????rpn???????????????????????????????????????????????????
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(base_feature, img_size, scale)

        self.log_info.append_log(f'\t rpn_locs:{rpn_locs.size()}, rpn_scores:{rpn_scores.size()}')
        self.log_info.append_log(f'\t rois:{rois.size()}, roi_indices:{roi_indices.size()}')

        rpn_loc_loss_all, rpn_cls_loss_all,roi_loc_loss_all,roi_cls_loss_all = 0,0,0,0

        temp_n = n
        for i in range(n):
            self.log_info.append_log(f' ')
            self.log_info.append_log(f'222 === batch_{i}: {labels[i]}: ########')

            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[roi_indices == i]
            feature = base_feature[i]

            self.log_info.append_log(f'\t bbox:{len(bbox)}, label:{len(label)}')
            self.log_info.append_log(f'\t rpn_loc:{rpn_loc.size()}, rpn_score:{rpn_score.size()}')
            self.log_info.append_log(f'\t roi:{roi.size()}, feature:{feature.size()}')

            # ?????????????????????????????????????????????????????????????????????
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor)
            gt_rpn_loc = torch.Tensor(gt_rpn_loc)
            gt_rpn_label = torch.Tensor(gt_rpn_label).long()

            self.log_info.append_log(f'\t gt_rpn_loc:{gt_rpn_loc.size()}, gt_rpn_label:{gt_rpn_label.size()}')

            if rpn_loc.is_cuda:
                gt_rpn_loc = gt_rpn_loc.cuda()
                gt_rpn_label = gt_rpn_label.cuda()
            # ?????????????????????????????????????????????????????????
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            self.log_info.append_log(f'\t rpn_loc_loss:{rpn_loc_loss}, rpn_cls_loss:{rpn_cls_loss}')

            # ?????????????????????????????????classifier??????????????????????????????
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalized_std)
            sample_roi = torch.Tensor(sample_roi)
            gt_roi_loc = torch.Tensor(gt_roi_loc)
            gt_roi_label = torch.Tensor(gt_roi_label).long()
            sample_roi_index = torch.zeros(len(sample_roi))

            self.log_info.append_log(
                f'\t sample_roi:{sample_roi.size()}, gt_roi_loc:{gt_roi_loc.size()}, gt_roi_label:{gt_roi_label.size()}')

            if feature.is_cuda:
                sample_roi = sample_roi.cuda()
                sample_roi_index = sample_roi_index.cuda()
                gt_roi_loc = gt_roi_loc.cuda()
                gt_roi_label = gt_roi_label.cuda()

            roi_cls_loc, roi_score = self.faster_rcnn.head(torch.unsqueeze(feature, 0), sample_roi,
                                                           sample_roi_index, img_size)

            self.log_info.append_log(f'\t roi_cls_loc:{roi_cls_loc.size()}, roi_score:{roi_score.size()}')

            #   ????????????????????????????????????????????????????????????
            n_sample = roi_cls_loc.size()[1]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            #   ????????????Classifier????????????????????????????????????
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)

            self.log_info.append_log(f'\t roi_loc_loss:{roi_loc_loss}, roi_cls_loss:{roi_cls_loss}')

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / temp_n, rpn_cls_loss_all / temp_n, roi_loc_loss_all / temp_n, roi_cls_loss_all / temp_n]
        losses = losses + [sum(losses)]

        self.log_info.append_log(f'losses:{losses}')

        return losses

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses[-1].backward()
        self.optimizer.step()
        return losses

