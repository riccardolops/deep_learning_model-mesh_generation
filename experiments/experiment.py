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

eps = 1e-10
class Experiment:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.epoch = 0
        self.metrics = {}

        filename = 'splits.json'

        if self.config.data_loader.labels is not None:
            num_classes = len(self.config.data_loader.labels)
        else:
            num_classes = 1
        if 'Jaccard' in self.config.loss.name or num_classes == 2:
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
        self.loss = LossFactory(self.config.loss.name, self.config.data_loader.labels)

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
        self.test_dataset = DatasetFactory(
            dataset_name=self.config.data_loader.dataset_name,
            root=self.config.data_loader.dataset,
            splits='test',
            transform=self.config.data_loader.preprocessing,
            # dist_map=['sparse', 'dense']
        ).get()

        # queue start loading when used, not when instantiated
        self.train_loader = self.train_dataset.get_loader(self.config.data_loader)
        self.val_loader = self.val_dataset.get_loader(self.config.data_loader)
        self.test_loader = self.test_dataset.get_loader(self.config.data_loader)


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
        volume = patch['data'][tio.DATA].float().cuda()
        if phase=='Test':
            gt = []
        else:
            gt = patch['dist'][tio.DATA].float().cuda()

        if 'Generation' in self.__class__.__name__:
            sparse = patch['sparse'][tio.DATA].float().cuda()
            images = torch.cat([volume, sparse], dim=1)
        else:
            images = volume

        emb_codes = torch.cat((
            patch[tio.LOCATION][:,:3],
            patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()

        return images, gt, emb_codes

    def train(self):

        self.model.train()
        self.evaluator.reset_eval()

        data_loader = self.train_loader

        losses = []
        for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
            images, gt, emb_codes = self.extract_data_from_patch(d)

            partition_weights = 1
            # TODO: Do only if not Competitor
            gt_count = torch.sum(gt, dim=list(range(1, gt.ndim)))
            if torch.sum(gt_count) == 0: continue
            partition_weights = (eps + gt_count) / torch.max(gt_count)

            self.optimizer.zero_grad()
            preds = self.model(images, emb_codes)

            assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
            loss = self.loss(preds, gt, partition_weights)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if self.config.experiment.name == 'Segmentation':
                preds = (preds > 0.5).squeeze().detach()
            else:
                preds = preds.squeeze().detach()

            gt = gt.squeeze()
            self.evaluator.compute_metrics(preds, gt)

        epoch_train_loss = sum(losses) / len(losses)
        epoch_metrics = self.evaluator.mean_metric()

        self.metrics['Train'] = {metric: value for metric, value in epoch_metrics.items()}

        log = {
            f'Epoch': self.epoch,
            f'Train/Loss': epoch_train_loss,
            f'Train/Lr': self.optimizer.param_groups[0]['lr']
        }
        for metric, value in epoch_metrics.items():
            log[f'Train/{metric}'] = value
        wandb.log(log)

        return epoch_train_loss


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
                aggregator = tio.inference.GridAggregator(sampler)
                gt_aggregator = tio.inference.GridAggregator(sampler)

                for j, patch in enumerate(loader):
                    images, gt, emb_codes = self.extract_data_from_patch(patch)

                    preds = self.model(images, emb_codes)
                    aggregator.add_batch(preds, patch[tio.LOCATION])
                    gt_aggregator.add_batch(gt, patch[tio.LOCATION])

                output = aggregator.get_output_tensor()
                gt = gt_aggregator.get_output_tensor()
                partition_weights = 1

                gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
                if torch.sum(gt_count) != 0:
                    partition_weights = (eps + gt_count) / (eps + torch.max(gt_count))

                loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
                losses.append(loss.item())

                output = output.squeeze(0)
                if self.config.experiment.name == 'Segmentation':
                    output = (output > 0.5)

                self.evaluator.compute_metrics(output, gt)

            epoch_loss = sum(losses) / len(losses)
            epoch_metrics = self.evaluator.mean_metric()

            log = {
                f'Epoch': self.epoch,
                f'{phase}/Loss': epoch_loss
            }
            for metric, value in epoch_metrics.items():
                log[f'{phase}/{metric}'] = value
            wandb.log(log)

            return epoch_metrics

    def predict(self, path_origin):
        # TODO: Redo but only for one image
        self.model.eval()
        with torch.inference_mode():
            subject_dict = {
                'data': self.config.data_loader.preprocessing(tio.ScalarImage(path_origin)),
                'dense': tio.LabelMap(path_origin),
            }
            subject = tio.Subject(**subject_dict)
            sampler = tio.inference.GridSampler(
                subject,
                self.config.data_loader.patch_shape,
                0
            )
            loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
            aggregator = tio.inference.GridAggregator(sampler)

            for j, patch in enumerate(loader):
                images, gt, emb_codes = self.extract_data_from_patch(patch, phase='Test')
                preds = self.model(images, emb_codes)
                aggregator.add_batch(preds, patch[tio.LOCATION])

            output = aggregator.get_output_tensor()

            output = output.squeeze(0)
            #output = (output > 0.5)
            output_img = nib.Nifti1Image(output.squeeze(0), nib.load(path_origin).affine, nib.load(path_origin).header)
            save_path = os.path.join(self.config.project_dir, self.config.title, 'output')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.basename(path_origin)
            path = os.path.join(save_path, filename)
            nib.save(output_img, path)

    def predict_series(self, phase, rnd=None):
        # TODO: Fix saving to correct dimensions
        self.model.eval()
        with torch.inference_mode():
            if phase == 'Train':
                dataset = self.train_dataset
            elif phase == 'Validation':
                dataset = self.val_dataset
            elif phase == 'Test':
                dataset = self.test_dataset

            if rnd==True:
                subject = random.choice(dataset)
                sampler = tio.inference.GridSampler(subject, self.config.data_loader.patch_shape, 0)
                loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                aggregator = tio.inference.GridAggregator(sampler)
                gt_aggregator = tio.inference.GridAggregator(sampler)
                img_aggregator = tio.inference.GridAggregator(sampler)

                for j, patch in enumerate(loader):
                    images, gt, emb_codes = self.extract_data_from_patch(patch, phase=phase)
                    preds = self.model(images, emb_codes)
                    img_aggregator.add_batch(images, patch[tio.LOCATION])
                    aggregator.add_batch(preds, patch[tio.LOCATION])
                    gt_aggregator.add_batch(gt, patch[tio.LOCATION])

                image = img_aggregator.get_output_tensor()
                output = aggregator.get_output_tensor()
                gt = gt_aggregator.get_output_tensor()
                gt = gt.squeeze(0)
                output = output.squeeze(0)
                output = (output > 0.5)
                image = image.squeeze(0)
                class_labels = {
                    0: "background",
                    1: "foreground"
                }
                wandb_out_logs = []
                wandb_img_logs = []

                image = image.numpy()
                output = output.numpy()
                gt = gt.numpy()

                for img_slice_no in range(min(image.shape)):
                    img = image[:, :, img_slice_no]
                    out = output[:, :, img_slice_no]
                    gtt = gt[:, :, img_slice_no]

                    # append the image to wandb_img_list to visualize
                    # the slices interactively in W&B dashboard
                    wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {img_slice_no}"))

                    # append the image and masks to wandb_mask_logs
                    # to see the masks overlayed on the original image
                    wandb_out_logs.append(wandb.Image(img, masks={
                        "predictions": {
                            "mask_data": out,
                            "class_labels": class_labels
                        },
                        "ground_truth": {
                            "mask_data": gtt,
                            "class_labels": class_labels
                        }
                    }))

                wandb.log({"Image": wandb_img_logs})
                wandb.log({"Output mask": wandb_out_logs})

            else:
                for i, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'Saving predictions.'):
                    sampler = tio.inference.GridSampler(
                        subject,
                        self.config.data_loader.patch_shape,
                        0
                    )
                    loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
                    aggregator = tio.inference.GridAggregator(sampler)

                    for j, patch in enumerate(loader):
                        images, gt, emb_codes = self.extract_data_from_patch(patch, phase='Test')
                        preds = self.model(images, emb_codes)
                        aggregator.add_batch(preds, patch[tio.LOCATION])

                    output = aggregator.get_output_tensor()

                    output = output.squeeze(0)
                    output = (output > 0.5)
                    path_origin = os.path.join(self.config.data_loader.dataset, subject.patient['image'][2:])
                    output_img = nib.Nifti1Image(output.squeeze(0), nib.load(path_origin).affine, nib.load(path_origin).header)
                    save_path = os.path.join(self.config.project_dir, self.config.title, 'output')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    path = os.path.join(save_path, (subject.patient['image'][11:]))
                    nib.save(output_img, path)