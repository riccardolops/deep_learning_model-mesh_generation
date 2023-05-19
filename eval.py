from statistics import mean
import torch
import pathlib
import numpy as np
from skimage import metrics
import os
import pandas as pd
import zipfile

class Eval:
    def __init__(self, config, skip_dump=False):
        self.list = []
        self.dic = {}
        self.config = config

    def reset_eval(self):
        self.list.clear()

    def compute_metrics(self, pred, gt):
        pred = pred.detach()
        gt = gt.detach()

        pred = pred.cuda()
        gt = gt.cuda()

        pred = pred[None, ...] if pred.ndim == 3 else pred
        gt = gt[None, ...] if gt.ndim == 3 else gt

        for evaluation_value in self.config.evaluation:
            if evaluation_value == 'IoU':
                self.iou(pred, gt)
            elif evaluation_value == 'Dice':
                self.dice(pred, gt)
            elif evaluation_value == 'MSE':
                self.mse(pred, gt)
            elif evaluation_value == 'MAE':
                self.mae(pred, gt)
            elif evaluation_value == 'R2':
                self.r2_score(pred, gt)
            else:
                raise Exception(f"Evaluation value {evaluation_value} can't be found.")

        self.list.append(self.dic.copy())
        self.dic.clear()

    def iou(self, pred, gt):
        eps = 1e-6
        pred = pred.to(torch.uint8)
        gt = gt.to(torch.uint8)
        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        iou_union = dice_union - intersection
        iou = (intersection + eps) / (iou_union + eps)
        self.dic['iou'] = iou.item()

    def dice(self, pred, gt):
        eps = 1e-6
        pred = pred.to(torch.uint8)
        gt = gt.to(torch.uint8)
        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        dice = (2 * intersection + eps) / (dice_union + eps)
        self.dic['dice'] = dice.item()

    def mse(self, pred, gt):
        self.dic['mse'] = torch.mean((pred - gt) ** 2)

    def mae(self, pred, gt):
        self.dic['mae'] = torch.mean(torch.abs(pred - gt))

    def r2_score(self, pred, gt):
        y_mean = torch.mean(gt)
        ss_total = torch.sum((gt - y_mean) ** 2)
        ss_residual = torch.sum((gt - pred) ** 2)
        self.dic['r2'] = 1 - (ss_residual / ss_total)


    def mean_metric(self):
        metrics = self.list[0].keys()
        mean_metrics = {metric: 0 for metric in metrics}
        for dictionary in self.list:
            for metric, value in dictionary.items():
                mean_metrics[metric] += value
        mean_metrics = {metric: value / len(self.list) for metric, value in mean_metrics.items()}
        self.reset_eval()
        return mean_metrics
