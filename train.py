#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 14:24
# @Author  : shiman
# @File    : train.py
# @describe:

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from frnn.utils.utils import get_classes
from frnn.frcnn_net import FasterRCNN
from frnn.utils.frcnn_training import weight_init
from frnn.utils.log_loss import LossHistory
from frnn.utils.dataloader import FRCNNDataset, frcnn_dataset_collate



if __name__ == '__main__':

    Cuda = False
    #
    classes_path = r'E:\ml_code\data\frcnn\voc_classes.txt'
    model_path = r'E:\ml_code\data\frcnn\voc_weights_resnet.pth'
    input_shape = [600,600]
    backbone = 'resnet50'
    pretrained = False
    anchors_size = [8,16,32]
    # 冻结阶段训练参数 （主干被冻结），占用显存较小，仅对网络进行微调
    init_epoch = 0
    freeze_epoch = 50
    freeze_batch_size = 4
    freeze_lr = 1e-4
    # 解冻阶段训练参数（主干不被冻结，backbone参数发生变化），占用显存较大，对网络所有参数进行调整
    unfreeze_epoch = 100
    unfreeze_batch_size = 2
    unfreeze_lr = 1e-5
    # 设置是否进行冻结训练
    freeze_train = True
    # 设置多线程读取数据
    num_workers = 0
    # 图片和标签路径
    train_annotation_path, val_annotation_path = '2007_train.txt', '2007_val.txt'
    # 获取标签名和数
    class_names, num_classes = get_classes(classes_path)
    #
    model = FasterRCNN(num_classes, mode='training',anchor_scales=anchors_size,backbone=backbone,
                       pretrained=pretrained)
    if not pretrained:
        weight_init(model)

    if model_path != '':
        print(f'load weights {model_path}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory('logs/')

    # 读取数据集对应txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    if True:
        batch_size = freeze_batch_size
        lr = freeze_lr
        start_epoch = init_epoch
        end_epoch = freeze_epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集')

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)

        gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         drop_last=True, pin_memory=True, collate_fn=frcnn_dataset_collate)

        gen_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                         drop_last=True, pin_memory=True, collate_fn=frcnn_dataset_collate)

        # 冻结一定部分训练
        if freeze_train:
            for param in model.extractor.parameters():
                param.requires_grad = False
        # 冻结bn层
        model.freeze_bn()

        train_util = FasterRCNNTrainer(model, optimizer)





