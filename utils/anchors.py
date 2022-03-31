#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 18:52
# @Author  : shiman
# @File    : anchors.py
# @describe:

import numpy as np

def generate_anchor_base(base_size=16, ratio=[0.5,1,2], anchor_scales=[8,16,32]):
    anchor_base = np.zeros((len(ratio)*len(anchor_scales), 4), dtype=np.float32)

    index = 0

    for i in ratio:
        for j in anchor_scales:
            h = base_size * j * np.sqrt(i)
            w = base_size * j * np.sqrt(1./i)

            anchor_base[index, 0] = -h / 2.
            anchor_base[index, 1] = -w /2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.

            index += 1

    return anchor_base

if __name__ == '__main__':

    anchor_base = generate_anchor_base()
    print(anchor_base)