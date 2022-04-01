#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 10:36
# @Author  : shiman
# @File    : dataset_annotation.py
# @describe:

import os
import random
import xml.etree.ElementTree as ET

from frnn.utils.utils import get_classes


# 准备工作收集待训练的数据集，利用labelimg制作Annotations
# 手动按照数据集中待分的种类生成文件classes_path

# 处理整个标签过程，包含ImageSets中的train\val\test.txt和训练用到的train\val.txt

classes_path = r'E:\ml_code\data\frcnn\voc_classes.txt'

trainval_percent = 0.9  # (训练集+验证集 ：测试集)
train_percent = 0.9     #  (训练集 ：验证集)

dataset_path = r'E:\ml_code\data\frcnn\VOC2007'
dataset_mark = [('2007', 'train'), ('2007', 'val')]
# 获取数据集分类
classes_name, _ = get_classes(classes_path)


def convert_annotation(image_id, list_file):
    in_file = open(f'{dataset_path}/Annotations/{image_id}.xml', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes_name or int(difficult) == 1:
            continue
        cls_id = classes_name.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(float(xml_box.find('xmin').text)), int(float(xml_box.find('ymin').text)),
             int(float(xml_box.find('xmax').text)), int(float(xml_box.find('ymax').text)))

        list_file.write(f' {",".join([str(a) for a in b])},{cls_id}')


def generate_index_txt():

    print('Generate txt in ImageSets')
    xml_file_path = os.path.join(dataset_path, 'Annotations')
    save_base_path = os.path.join(dataset_path, 'ImageSets/Main')
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    # 获得所有xml文件
    temp_xml = os.listdir(xml_file_path)
    total_xml = [i for i in temp_xml if i.endswith('.xml')]

    num_xml = len(total_xml)
    # 训练验证数据集总个数， 训练数据集个数
    tv = int(num_xml * trainval_percent)
    tr = int(tv * train_percent)
    trainval_v = random.sample(range(num_xml), tv)
    train_v = random.sample(trainval_v, tr)

    #
    print(f'total size:{num_xml},'
          f'train and val size:{tv}, test size:{num_xml - tv}, '
          f'train size:{tr}, val size:{tv - tr}')
    #
    f_trainval = open(os.path.join(save_base_path, 'trainval.txt'), 'w')
    f_test = open(os.path.join(save_base_path, 'test.txt'), 'w')
    f_train = open(os.path.join(save_base_path, 'train.txt'), 'w')
    f_val = open(os.path.join(save_base_path, 'val.txt'), 'w')

    for i in range(num_xml):
        name = total_xml[i][:-4] + '\n'
        if i in trainval_v:
            f_trainval.write(name)
            if i in train_v:
                f_train.write(name)
            else:
                f_val.write(name)
        else:
            f_test.write(name)
    f_train.close(), f_test.close(), f_trainval.close(), f_val.close()

    print('Generate txt in ImageSets done')


def generate_train_file():

    print('Generate 2007_train.txt and 2007_val.txt for train')

    for year, image_set in dataset_mark:
        image_ids = open(os.path.join(dataset_path, f'ImageSets/Main/{image_set}.txt'),
                         encoding='utf-8').read().strip().split()
        with open(f'{year}_{image_set}.txt', 'w', encoding='utf-8') as f:

            for image_id in image_ids:
                t_img_file = f'{dataset_path}/JPEGImages/{image_id}.jpg'
                f.write(t_img_file)

                convert_annotation(image_id, f)
                f.write('\n')

    print('Generate 2007_train.txt and 2007_val.txt for train done')



if __name__ == '__main__':

    random.seed(816)

    generate_index_txt()

    generate_train_file()

















