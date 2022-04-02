#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 13:48
# @Author  : shiman
# @File    : utils.py
# @describe:

import numpy as np
from PIL import Image


def get_classes(classes_path):
    """获取分类名和分类数"""
    with open(classes_path, encoding='utf-8') as f:
        class_name = f.readlines()
    class_names = [c.strip() for c in class_name]
    return class_names, len(class_names)


def get_new_img_size(src_h, src_w, img_min_side=600):
    """将短边重采样成 img_min_side大小，获得新的尺寸"""
    if src_h < src_w:
        f = float(img_min_side) / src_h
        r_w = int(f * src_w)
        r_h = int(img_min_side)
    else:
        f = float(img_min_side) / src_w
        r_h = int(f * src_h)
        r_w = int(img_min_side)

    return r_h, r_w


def resize_image(image, size):
    h, w = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


def cvtColor(image):
    """将所有图片转换成RGB"""
    if len(np.shape(image)) == 3 and np.shape(image)[-1] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
