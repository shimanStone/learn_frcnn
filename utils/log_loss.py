#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 15:10
# @Author  : shiman
# @File    : log_loss.py
# @describe:

import os
import scipy.signal
import matplotlib.pyplot as plt


class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = f'{self.log_dir}/loss_{self.time_str}'
        self.losses = []
        self.val_losses = []

        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        with open(f'{self.save_path}/epoch_loss_{str(self.time_str)}.txt', 'a') as f:
            f.write(str(loss))
            f.write('/n')
        with open(f'{self.save_path}/epoch_val_loss_{str(self.time_str)}.txt', 'a') as f:
            f.write(str(val_loss))
            f.write('/n')
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth=2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 25

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3),
                     'green', linestype='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3),
                     '8B4513', linestype='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(f'{self.save_path}/epoch_loss_{str(self.time_str)}.png')

        plt.cla()
        plt.close('all')

