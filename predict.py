#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 13:36
# @Author  : shiman
# @File    : predict.py
# @describe

import os
from fastercnn.frcnn import FRCNN
from PIL import Image


if __name__ == '__main__':

    # FRCNN._defaults['model_path'] = r'E:\ml_code\data\frcnn\voc_weights_vgg.pth'
    # FRCNN._defaults['backbone'] = r'vgg16'
    FRCNN.set_defaults('model_path', r'E:\ml_code\data\frcnn\voc_weights_vgg.pth')
    FRCNN.set_defaults('backbone', 'vgg16')
    frcc = FRCNN()
    mode='predict'
    dir_origin_path, dir_save_path = '',''
    img = r'E:\ml_code\data\frcnn\street.jpg'

    image = Image.open(img)
    r_image = frcc.detect_image(image, crop=False)
    r_image.show()
