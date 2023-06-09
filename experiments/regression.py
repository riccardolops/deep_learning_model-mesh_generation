import sys
import os
import argparse
import logging
import logging.config
import yaml
import pathlib
import builtins
import socket
import time
import random
import numpy as np
import torch
import logging
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data
import wandb
import nibabel as nib

from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.DatasetFactory import DatasetFactory
from dataloader.AugFactory import AugFactory
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from eval import Eval as Evaluator
from utils.utils import Approximated_Heaviside

eps = 1e-10
class Regression:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        num_classes = len(self.config.data_loader.labels)
        if 'Jaccard' in self.config.loss_seg.name or num_classes == 2:
            num_classes = 1

        # load model
        model_name = self.config.model.name
        in_ch = 2 if self.config.experiment.name == 'Generation' else 1
        emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

        self.model = ModelFactory(model_name, num_classes, in_ch, emb_shape).get().cuda()
        self.model = nn.DataParallel(self.model)
        wandb.watch(self.model, log_freq=10)

        # load optimizer
        optim_name = self.config.optimizer.name
        train_params = self.model.parameters()
        lr = self.config.optimizer.learning_rate

        self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

        # load scheduler
        sched_name = self.config.lr_scheduler.name
        sched_milestones = self.config.lr_scheduler.get('milestones', None)
        sched_gamma = self.config.lr_scheduler.get('factor', None)


        self.scheduler = SchedulerFactory(
            sched_name,
            self.optimizer,
            milestones=sched_milestones,
            gamma=sched_gamma,
            mode='max',
            verbose=True,
            patience=15
        ).get()

        # load loss
        self.loss = LossFactory(self.config.loss_seg.name, self.config.loss_reg.name, self.config.data_loader.labels)

        # load evaluator
        self.evaluator = Evaluator(self.config, skip_dump=True)

        self.train_dataset = DatasetFactory(
            dataset_name=self.config.data_loader.dataset_name,
            root=self.config.data_loader.dataset,
            splits='train',
            transform=tio.Compose([
                tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
                self.config.data_loader.preprocessing,
                self.config.data_loader.augmentations,
                ]),
            # dist_map=['sparse','dense']
        ).get()
        self.val_dataset = DatasetFactory(
            dataset_name=self.config.data_loader.dataset_name,
            root=self.config.data_loader.dataset,
            splits='val',
            transform=self.config.data_loader.preprocessing,
            # dist_map=['sparse', 'dense']
        ).get()

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)


        if self.config.trainer.reload:
            self.load()

    def save(self, name):
        if '.pth' not in name:
            name = name + '.pth'
        path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
        logging.info(f'Saving checkpoint at {path}')
        state = {
            'title': self.config.title,
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, path)

    def load(self):
        path = self.config.trainer.checkpoint
        logging.info(f'Loading checkpoint from {path}')
        state = torch.load(path)

        if 'title' in state.keys():
            # check that the title headers (without the hash) is the same
            self_title_header = self.config.title[:-11]
            load_title_header = state['title'][:-11]
            if self_title_header == load_title_header:
                self.config.title = state['title']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['state_dict'])
        self.epoch = state['epoch'] + 1

        if 'metrics' in state.keys():
            self.metrics = state['metrics']

    def extract_data_from_patch(self, patch, phase=None):
        images = patch['data'][tio.DATA].float().cuda()
        if phase=='Test':
            gt_seg = []
            gt_dis = []
        else:
            gt_seg = patch['dense'][tio.DATA].float().cuda()
            gt_dis = patch['distance'][tio.DATA].float().cuda()

        emb_codes = torch.cat((
            patch[tio.LOCATION][:,:3],
            patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()

        return images, gt_seg, gt_dis, emb_codes

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader

        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt_seg, gt_dis, emb_codes = self.extract_data_from_patch(d)

            partition_weights = 1
            # TODO: Do only if not Competitor
            gt_count = torch.sum(gt_seg == 1, dim=list(range(1, gt_seg.ndim)))
            if torch.sum(gt_count) == 0: continue
            partition_weights = (eps + gt_count) / torch.max(gt_count)

            self.optimizer.zero_grad()
            preds_seg, preds_dis = self.model(images, emb_codes)

            approx_heaviside = Approximated_Heaviside()

            preds_seg = approx_heaviside(preds_dis)

            assert preds_seg.ndim == gt_seg.ndim, f'Gt of segmentation and output dimensions are not the same before loss. {preds_seg.ndim} vs {gt_seg.ndim}'
            assert preds_dis.ndim == gt_dis.ndim, f'Gt of distance and output dimensions are not the same before loss. {preds_dis.ndim} vs {gt_dis.ndim}'
            loss = self.loss(preds_seg, gt_seg, preds_dis, gt_dis, partition_weights)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            preds_seg = (preds_seg > 0.5).squeeze().detach()

            gt_seg = gt_seg.squeeze()
            self.evaluator.compute_metrics(preds_seg, gt_seg)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_iou, epoch_dice = self.evaluator.mean_metric(phase='Train')

        self.metrics['Train'] = {
            'iou': epoch_iou,
            'dice': epoch_dice,
        }

        wandb.log({
            f'Epoch': self.epoch,
            f'Train/Loss': epoch_train_loss,
            f'Train/Dice': epoch_dice,
            f'Train/IoU': epoch_iou,
            f'Train/Lr': self.optimizer.param_groups[0]['lr']
        })

        return epoch_train_loss, epoch_iou

    def test(self, phase):

        self.model.eval()

        # with torch.no_grad():
        with torch.inference_mode():
            self.evaluator.reset_eval()
            losses = []

            if phase == 'Test':
                dataset = self.test_dataset
            elif phase == 'Validation':
                dataset = self.val_dataset

            for i, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'{phase} epoch {str(self.epoch)}'):

                sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        0
                )
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator_seg = tio.inference.GridAggregator(sampler)
                aggregator_dis = tio.inference.GridAggregator(sampler)
                gt_seg_aggregator = tio.inference.GridAggregator(sampler)
                gt_dis_aggregator = tio.inference.GridAggregator(sampler)

                for j, patch in enumerate(loader):
                    images, gt_seg, gt_dis, emb_codes = self.extract_data_from_patch(patch)

                    preds_seg, preds_dis = self.model(images, emb_codes)
                    approx_heaviside = Approximated_Heaviside()

                    preds_seg = approx_heaviside(preds_dis)
                    aggregator_seg.add_batch(preds_seg, patch[tio.LOCATION])
                    aggregator_dis.add_batch(preds_dis, patch[tio.LOCATION])
                    gt_seg_aggregator.add_batch(gt_seg, patch[tio.LOCATION])
                    gt_dis_aggregator.add_batch(gt_dis, patch[tio.LOCATION])

                output_seg = aggregator_seg.get_output_tensor()
                output_dis = aggregator_dis.get_output_tensor()
                gt_seg = gt_seg_aggregator.get_output_tensor()
                gt_dis = gt_dis_aggregator.get_output_tensor()
                partition_weights = 1

                gt_count = torch.sum(gt_seg == 1, dim=list(range(1, gt_seg.ndim)))
                if torch.sum(gt_count) != 0:
                    partition_weights = (eps + gt_count) / (eps + torch.max(gt_count))

                loss = self.loss(output_seg.unsqueeze(0), gt_seg.unsqueeze(0), output_dis.unsqueeze(0), gt_dis.unsqueeze(0), partition_weights)
                losses.append(loss.item())

                output_seg = output_seg.squeeze(0)
                output_seg = (output_seg > 0.5)

                self.evaluator.compute_metrics(output_seg, gt_seg)

            epoch_loss = sum(losses) / len(losses)
            epoch_iou, epoch_dice = self.evaluator.mean_metric(phase=phase)

            wandb.log({
                f'Epoch': self.epoch,
                f'{phase}/Loss': epoch_loss,
                f'{phase}/Dice': epoch_dice,
                f'{phase}/IoU': epoch_iou
            })

            return epoch_iou, epoch_dice

    def predict(self, path_origin):
        # TODO: Redo but only for one image
        self.model.eval()
        with torch.inference_mode():
            subject_dict = {
                'data': self.config.data_loader.preprocessing(tio.ScalarImage(path_origin)),
                'dense': tio.LabelMap(path_origin),
                'distance': tio.ScalarImage(path_origin),
            }
            subject = tio.Subject(**subject_dict)
            sampler = tio.inference.GridSampler(
                subject,
                self.config.data_loader.patch_shape,
                0
            )
            loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
            aggregator_seg = tio.inference.GridAggregator(sampler)
            aggregator_dis = tio.inference.GridAggregator(sampler)

            for j, patch in enumerate(loader):
                images, gt_seg, gt_dis, emb_codes = self.extract_data_from_patch(patch, phase='Test')
                preds_seg, preds_dis = self.model(images, emb_codes)
                aggregator_seg.add_batch(preds_seg, patch[tio.LOCATION])
                aggregator_dis.add_batch(preds_dis, patch[tio.LOCATION])

            output_seg = aggregator_seg.get_output_tensor()
            output_dis = aggregator_dis.get_output_tensor()

            output_seg = output_seg.squeeze(0)
            output_dis = output_dis.squeeze(0)
            output_seg = (output_seg > 0.5)
            output_im_seg = nib.Nifti1Image(output_seg.squeeze(0), nib.load(path_origin).affine, nib.load(path_origin).header)
            output_im_dis = nib.Nifti1Image(output_dis.squeeze(0), nib.load(path_origin).affine, nib.load(path_origin).header)
            save_path = os.path.join(self.config.project_dir, self.config.title, 'output')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.basename(path_origin)
            path = os.path.join(save_path, 'seg' + filename)
            print(f'Saving to: {path}')
            nib.save(output_im_seg, path)
            path = os.path.join(save_path, 'dis' + filename)
            print(f'Saving to: {path}')
            nib.save(output_im_dis, path)