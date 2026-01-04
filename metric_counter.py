import logging
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter

WINDOW_SIZE = 100


class MetricCounter:
    def __init__(self, exp_name):
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename='{}.log'.format(exp_name), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0

    def add_image(self, x: np.ndarray, tag: str):
        self.images[tag].append(x)

    def clear(self):
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)

    def add_losses(self, l_G):
        for name, value in zip(('G_loss', None), (l_G, None)):
            self.metrics[name].append(value)

    def train_add_losses(self, l_G, loss, sub_loss):
        for name, value in zip(('G_loss', 'loss', 'sub_loss'), (l_G, loss, sub_loss)):
            self.metrics[name].append(value)

    def val_add_losses(self, l_G):
        for name, value in zip(('G_loss', None), (l_G, None)):
            self.metrics[name].append(value)

    def loss_add_losses(self, l_G):
        for name, value in zip(('loss', None), (l_G, None)):
            self.metrics[name].append(value)
    def sub_loss_add_losses(self, l_G):
        for name, value in zip(('sub_loss', None), (l_G, None)):
            self.metrics[name].append(value)

    def add_metrics(self, psnr, ssim):
        for name, value in zip(('PSNR', 'SSIM'),
                               (psnr, ssim)):
            self.metrics[name].append(value)

    def train_loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ('G_loss', 'loss', 'sub_loss', 'PSNR', 'SSIM'))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def val_loss_message(self):
        metrics = ((k, np.mean(self.metrics[k][-WINDOW_SIZE:])) for k in ('G_loss', 'PSNR', 'SSIM'))
        return '; '.join(map(lambda x: f'{x[0]}={x[1]:.4f}', metrics))

    def train_write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'Train'
        for tag in ('G_loss', 'loss', 'sub_loss', 'SSIM', 'PSNR'):
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []

        return self.metrics['G_loss'], self.metrics['loss'], self.metrics['sub_loss'], self.metrics['SSIM'], self.metrics['PSNR']
    def val_write_to_tensorboard(self, epoch_num, validation=False):
        scalar_prefix = 'Validation' if validation else 'test'
        for tag in ('G_loss', 'SSIM', 'PSNR'):
            self.writer.add_scalar(f'{scalar_prefix}_{tag}', np.mean(self.metrics[tag]), global_step=epoch_num)
        for tag in self.images:
            imgs = self.images[tag]
            if imgs:
                imgs = np.array(imgs)
                self.writer.add_images(tag, imgs[:, :, :, ::-1].astype('float32') / 255, dataformats='NHWC',
                                       global_step=epoch_num)
                self.images[tag] = []

        return self.metrics['G_loss'], self.metrics['SSIM'], self.metrics['PSNR']

    def update_best_model(self):
        cur_metric = np.mean(self.metrics['PSNR'])
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
