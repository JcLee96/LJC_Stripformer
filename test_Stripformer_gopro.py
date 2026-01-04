import cv2
import logging
import numpy as np
import os
import pdb
import random
import re
import torch
import torch.optim as optim
import torchvision
import tqdm
import yaml
from functools import partial
from joblib import cpu_count
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models2 import get_model
from models.networks import get_nets

cv2.setNumThreads(0)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import csv

class Trainer:
    def __init__(self, config, train: DataLoader, val: DataLoader, test: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.metric_counter = MetricCounter(config['experiment_desc'])

    def train(self):
        self._init_params()
        start_epoch = 0

        self.scheduler_G.step()
        training_state = (torch.load('./best_model/best.pth'))
        new_weight = self.netG.state_dict()
        new_weight.update(training_state)
        self.netG.load_state_dict(new_weight)
        self._test()

    def _test(self):
        csv_path = "./GRU_Stripformer4.csv"
        self.metric_counter.clear()
        epoch_size = len(self.test_dataset)
        tq = tqdm.tqdm(self.test_dataset, total=epoch_size)
        tq.set_description('test')
        i = 0
        with open(csv_path, mode='w', newline='') as wf:
            writer = csv.writer(wf)

            # 결과 CSV 파일에 헤더 행 쓰기
            writer.writerow(['path', 'psnr', 'ssim', 'min', 'max', 'avg', 'std'])
            for data in tq:
                with torch.no_grad():
                    inputs, targets, sub, sub_target, data_name = self.model.get_input(data)
                    outputs = self.netG(inputs, sub, return_subframe=False)

                    loss_G = self.criterionG(outputs, targets, inputs, sub, sub_target, sub_img=False)
                    self.metric_counter.val_add_losses(loss_G.item())
                    curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(inputs, outputs, targets)

                    min = torch.min(targets*255)
                    max = torch.max(targets*255)
                    avg = torch.mean(targets*255)
                    std = torch.std(targets*255)

                    writer.writerow([data_name[0].split("/")[-1], curr_psnr, curr_ssim, min.item(), max.item(), avg.item(), std.item()])

                    self.metric_counter.add_metrics(curr_psnr, curr_ssim)
                    if not i:
                        self.metric_counter.add_image(img_for_vis, tag='test')
                    i += 1
                    if i > epoch_size:
                        break
            tq.close()
            G_loss, P, S = self.metric_counter.val_write_to_tensorboard(1, validation=True)
            wandb.log({"test_G_loss": np.mean(G_loss),
                       "test_PSNR": np.mean(P),
                       "test_SSIM": np.mean(S)})


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    wandb.login(key='e476bb91d83495c4174473429b20f197416f08c9')
    wandb.init(
        project="Stripformer",
        entity="ljc",
        job_type="train",
        name='Stripformer_test_290_best',
    )


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
