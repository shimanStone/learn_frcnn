#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 13:36
# @Author  : shiman
# @File    : predict.py
# @describe

from frnn.frcnn import FRCNN
from PIL import Image


if __name__ == '__main__':

    frcc = FRCNN()
    mode='predict'
    dir_origin_path, dir_save_path = '',''
    img = r'E:\ml_code\frnn\data\street.jpg'

    image = Image.open(img)
    r_image = frcc.detect_image(image, crop=False)
    r_image.show()
