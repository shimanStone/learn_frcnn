#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 13:42
# @Author  : shiman
# @File    : frcnn.py
# @describe: faster-rcnn 模型


import os
import numpy as np
import torch
import torch.nn as nn
import colorsys
from PIL import ImageDraw, ImageFont

from frnn.net.frcnn_net import FasterRCNN

from frnn.utils.utils_bbox import DecodeBox
from frnn.utils.utils import get_classes, get_new_img_size, resize_image, \
                                cvtColor, preprocess_input


class FRCNN(object):
    _defaults = {
        'model_path': '../data/frcnn/voc_weights_resnet.pth',  # 训练模型
        'classes_path': '../data/frcnn/voc_classes.txt',  # 分类
        'backbone': 'resnet50',  # 主干特征提取网络
        'confidence': 0.5,  # 置信度
        'nms_iou': 0.3,  # 非极大抑制
        'anchor_size': [8, 16, 32],  # 指定先验框大小
        'cuda': False,  # 服务器类型
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return f'Unrecognized attribute name {n}'

    def __init__(self, **kwargs):

        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  # 设置属性值
        # 获得种类及种类数
        self.class_names, self.num_classes = get_classes(self.classes_path)
        #  shape : 1, num_classes+1
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)
        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #
        self.generate()

    def generate(self):
        # 载入模型和权重
        self.net = FasterRCNN(self.num_classes, 'predict',
                              anchor_scales=self.anchor_size, backbone=self.backbone)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model, anchors, and classes loaded')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image, crop=False):

        # 获取输入图像高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 计算resize后的大小，resize的短边为600
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #
        image = cvtColor(image)
        # image resize
        image_data = resize_image(image, input_shape)
        # 添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # 框调整参数，框得分，框坐标
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            # 利用classifer预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape,
                                             input_shape, nms_iou=self.nms_iou, confidence=self.confidence)
            # 如果没有检测出物体，则返回原图
            if len(results[0]) <= 0:
                return image
            # 标签，置信度，位置
            top_labels = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]
            # 设置字体与边框厚度
            font = ImageFont.truetype(font='./data/simhei.ttf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

            # 是否对目标进行裁剪
            if crop:
                for i, c in list(enumerate(top_labels)):
                    top, left, bottom, right = top_boxes[i]
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                    right = min(image.size[0], np.floor(right).astype('int32'))

                    dir_save_path = 'img_crop'
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    crop_image = image.crop([left, top, right, bottom])
                    crop_image.save(f'{dir_save_path}/crop_{i}.png', quality=95, subsampling=0)
                    print(f'save crop_{i}.png to {dir_save_path}')

            # 图像绘制
            for i, c in list(enumerate(top_labels)):
                predicted_class = self.class_names[int(c)]
                box = top_boxes[i]
                score = top_conf[i]

                top, left, bottom, right = box

                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                # print(label, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

            return image