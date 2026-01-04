import logging
import os
import random
from functools import partial

import cv2
import numpy as np
import torch
import torch.optim as optim
import tqdm
import wandb
import yaml
from joblib import cpu_count
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import PairedDataset
from metric_counter import MetricCounter
from models.key_mapping import auto_map_module_names
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets

cv2.setNumThreads(0)


class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader, test: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.best_model_path = config['model']['best_model_path']

    def train(self):
        self._init_params()
        start_epoch = 0
        if os.path.exists('{0}/last_Stripformer_gopro.pth'.format(self.best_model_path)):
            print('load_pretrained')
            training_state = (torch.load('{0}/last_Stripformer_gopro.pth'.format(self.best_model_path)))
            start_epoch = training_state['epoch']
            new_weight = self.netG.state_dict()
            new_weight.update(training_state['model_state'])
            self.netG.load_state_dict(new_weight)
            new_optimizer = self.optimizer_G.state_dict()
            new_optimizer.update(training_state['optimizer_state'])
            self.optimizer_G.load_state_dict(new_optimizer)
            new_scheduler = self.scheduler_G.state_dict()
            new_scheduler.update(training_state['scheduler_state'])
            self.scheduler_G.load_state_dict(new_scheduler)
        else:
            print('load_GoPro_pretrained')
            training_state = (torch.load('./Pre_trained_model/Stripformer_gopro.pth'))
            new_weight = self.netG.state_dict()
            new_weight.update(training_state)
            self.netG = auto_map_module_names(training_state, new_weight, self.netG)


        print('total_epoch:', config['num_epochs'])
        for epoch in range(start_epoch, config['num_epochs']):
            self._run_epoch(epoch)
            self._validate(epoch)
            # if epoch % 30 == 0 or epoch == (config['num_epochs']-1):
            #     self._validate(epoch)
            self.scheduler_G.step()

            scheduler_state = self.scheduler_G.state_dict()
            training_state = {'epoch': epoch,  'model_state': self.netG.state_dict(),
                              'scheduler_state': scheduler_state, 'optimizer_state': self.optimizer_G.state_dict()}
            if self.metric_counter.update_best_model():
                torch.save(training_state['model_state'], './{0}/best_{1}.pth'.format(self.best_model_path, epoch))
                torch.save(training_state['model_state'], './{0}/best.pth'.format(self.best_model_path))

            # if epoch % 200 == 0:
            #     torch.save(training_state, 'last_{}_{}.pth'.format(self.config['experiment_desc'], epoch))

            if epoch == (config['num_epochs']-1):
                torch.save(training_state['model_state'], './{0}/final_{1}.pth'.format(self.best_model_path, self.config['experiment_desc']))

            torch.save(training_state, './{0}/last_{1}.pth'.format(self.best_model_path, self.config['experiment_desc']))
            # logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
            #     self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))


        training_state = (torch.load('./{0}/best.pth'.format(self.best_model_path)))
        new_weight = self.netG.state_dict()
        new_weight.update(training_state)
        self.netG.load_state_dict(new_weight)

        self._test()

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        # epoch_size = config.get('train_batches_per_epoch') or len(self.train_dataset)
        epoch_size = config.get('train_batches_per_epoch')
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            inputs, targets, sub, sub_target = self.model.get_input(data)

            outputs, predict_sub = self.netG(inputs, sub, return_subframe=True)
            self.optimizer_G.zero_grad()
            loss, sub_loss = self.criterionG(outputs, targets, inputs, predict_sub, sub_target, sub_img=True)
            loss_G = loss + sub_loss
            loss_G.backward()
            self.optimizer_G.step()
            self.metric_counter.train_add_losses(loss_G.item(), loss_G.item(), sub_loss.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.train_loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.train_write_to_tensorboard(epoch)


    def _validate(self, epoch):
        self.metric_counter.clear()
        # epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        epoch_size = config.get('val_batches_per_epoch')
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            with torch.no_grad():
                inputs, targets, sub, sub_target = self.model.get_input(data)
                outputs = self.netG(inputs, sub, return_subframe=False)
                loss_G = self.criterionG(outputs, targets, inputs, sub, sub_target, sub_img=False)
                self.metric_counter.val_add_losses(loss_G.item())
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                tq.set_postfix(loss=self.metric_counter.val_loss_message())
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='val')
                i += 1
                if i > epoch_size:
                    break
        tq.close()
        self.metric_counter.val_write_to_tensorboard(epoch, validation=True)

    def _test(self):
        self.metric_counter.clear()
        # epoch_size = config.get('test_batches_per_epoch') or len(self.val_dataset)
        epoch_size = len(self.test_dataset)
        tq = tqdm.tqdm(self.test_dataset, total=epoch_size)
        tq.set_description('test')
        i = 0
        for data in tq:
            with torch.no_grad():
                inputs, targets, sub, sub_target = self.model.get_input(data)
                outputs = self.netG(inputs, sub, return_subframe=False)
                loss_G = self.criterionG(outputs, targets, inputs, sub, sub_target, sub_img=False)
                self.metric_counter.val_add_losses(loss_G.item())
                curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)
                self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                if not i:
                    self.metric_counter.add_image(img_for_vis, tag='test')
                i += 1
                if i > epoch_size:
                    break
        tq.close()
        self.metric_counter.val_write_to_tensorboard(1, validation=True)



    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'], eta_min=self.config['scheduler']['min_lr'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    def _init_params(self):
        self.criterionG = get_loss(self.config['model'])
        self.netG = get_nets(self.config['model'])
        self.netG.cuda()
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.scheduler_G = self._get_scheduler(self.optimizer_G)


if __name__ == '__main__':
    with open('config/config_Stripformer_gopro.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # set random seed
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    batch_size = config.pop('batch_size')
    train_get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=True, drop_last=False)
    val_get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(), shuffle=False, drop_last=False)

    train_datasets = map(config.pop, ('train', 'val1'))
    train_datasets = map(PairedDataset.from_config, train_datasets)

    val_datasets = map(config.pop, ('val2', 'test'))
    val_datasets = map(PairedDataset.from_config, val_datasets)

    train, _ = map(train_get_dataloader, train_datasets)
    val, test = map(val_get_dataloader, val_datasets)

    trainer = Trainer(config, train=train, val=val, test=test)

    trainer.train()

    wandb.finish()
