#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 15:55
# @Author  : shiman
# @File    : dataloader.py
# @describe:

import cv2
import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset

from frnn.utils.utils import cvtColor
from frnn.utils.utils import preprocess_input


class FRCNNDataset_(Dataset):

    def __init__(self, annotation_lines, input_shape=[600,600], train=True):

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # 训练时进行数据的随机增强，验证时不进行
        image, y = self.get_random_data(self.annotation_lines[index], self.input_shape[:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2,0,1))
        box_data = np.zeros((len(y), 5))
        if len(y)>0:
            box_data[:len(y)] = y

        box = box_data[:, :4]
        label = box_data[:, -1]

        return image, box, label

    def rand(self, a=0., b=1.):
        return np.random.rand()*(b-a)+a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        #
        image = Image.open(line[0])
        image = cvtColor(image)
        #
        iw, ih = image.size
        h, w = input_shape
        # 获得预测框
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # 验证数据集重采样，annotation框修改
            # 输入影像重采样成 长边为600的，再将重采样后的影像填充进600*600的一个新影像中，多余部分加入灰度条
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image,(dx,dy))

            image_data = np.array(new_image, np.float32)

            # 对真实框进行调整
            if len(box) >0:
                np.random.shuffle(box)
                box[:,[0,2]] = box[:,[0,2]]*nw/iw + dx
                box[:,[1,3]] = box[:,[1,3]]*nh/ih + dy
                box[:,[0,1]][box[:,[0,1]] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w, bow_h = box[:,2] - box[:,0], box[:,3] - box[:,1]
                box = box[np.logical_and(box_w>1,bow_h>1)]

            return image_data, box

        # 训练数据集，对图像进行缩放并进行长、宽扭曲
        new_ar = w/h * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(.25,2)
        if new_ar <1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        dx = int(self.rand(0,w-nw))
        dy = int(self.rand(0,h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx,dy))
        image = new_image

        # 左右翻转
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1,sat) if self.rand()<.5 else 1/self.rand(1,sat)
        val = self.rand(1,val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[...,0] += hue*360
        x[...,0][x[...,0]>1] -= 1
        x[...,0][x[...,0]<0] += 1
        x[...,1] *= sat
        x[...,2] *= val
        x[x[:,:,0]>360,0] =360
        x[:,:,1:][x[:,:,1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) *255

        # 对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:,[0,2]] = w - box[:,[2,0]]
            box[:, [0, 1]][box[:, [0, 1]] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w, bow_h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, bow_h > 1)]

        return image_data, box


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        image, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box = box_data[:, :4]
        label = box_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        #   读取图像并转换成RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)
        #   获得图像的高宽与目标高宽
        iw, ih = image.size
        h, w = input_shape
        #   获得预测框
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #   翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #   色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        #   对真实框进行调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

def frcnn_dataset_collate(batch):
    """DataLoader中collate_fn使用"""
    images, bboxes, labels = [], [], []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)

    return images, bboxes, labels




