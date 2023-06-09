import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from .DiceLoss import DiceLoss
from .JaccardLoss import JaccardLoss
from .CrossEntropyLoss import CrossEntropyLoss
from .BCEWithLogitsLoss import BCEWithLogitsLoss
from .BoundaryLoss import BoundaryLoss
from .MSELoss import MSELoss
from .L1_LproductLoss import L1_LproductLoss

class LossFactory:
    def __init__(self, names_seg, names_reg, classes, weights=None):
        self.names_seg = names_seg
        self.names_reg = names_reg
        if not isinstance(self.names_seg, list):
            self.names_seg = [self.names_seg]
        if not isinstance(self.names_reg, list):
            self.names_reg = [self.names_reg]

        print(f'Losses used for segmentation output: {self.names_seg}')
        print(f'Losses used for regression output: {self.names_reg}')
        self.classes = classes
        self.weights = weights
        self.losses_seg = {}
        self.losses_reg = {}
        for name_seg in self.names_seg:
            loss = self.get_loss(name_seg)
            self.losses_seg[name_seg] = loss
        for name_reg in self.names_reg:
            loss = self.get_loss(name_reg)
            self.losses_seg[name_reg] = loss

    def get_loss(self, name):
        if name == 'CrossEntropyLoss':
            loss_fn = CrossEntropyLoss(self.weights, True)
        elif name == 'BCEWithLogitsLoss':
            loss_fn = BCEWithLogitsLoss(self.weights)
        elif name == 'Jaccard':
            loss_fn = JaccardLoss(weight=self.weights)
        elif name == 'DiceLoss':
            loss_fn = DiceLoss(self.classes)
        elif name == 'BoundaryLoss':
            loss_fn = BoundaryLoss()
        elif name == 'MSE':
            loss_fn = MSELoss()
        elif name == 'L1_Lp':
            loss_fn = L1_LproductLoss()
        else:
            raise Exception(f"Loss function {name} can't be found.")

        return loss_fn

    def __call__(self, pred_seg, gt_seg, pred_reg, gt_reg, partition_weights):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        assert pred_seg.device == gt_seg.device
        assert pred_reg.device == gt_reg.device
        assert gt_seg.device != 'cpu'
        assert gt_reg.device != 'cpu'

        cur_loss = []
        for loss_name in self.losses_seg.keys():
            loss = self.losses_seg[loss_name](pred_seg, gt_seg)
            if torch.isnan(loss.sum()):
                raise ValueError(f'Loss {loss_name} has some NaN')
            # print(f'Loss {self.losses[loss_name].__class__.__name__}: {loss}')
            loss = loss * partition_weights
            cur_loss.append(loss.mean())
        for loss_name in self.losses_reg.keys():
            loss = self.losses_reg[loss_name](pred_reg, gt_reg)
            if torch.isnan(loss.sum()):
                raise ValueError(f'Loss {loss_name} has some NaN')
            # print(f'Loss {self.losses[loss_name].__class__.__name__}: {loss}')
            loss = loss * partition_weights
            cur_loss.append(loss.mean())
        return torch.sum(torch.stack(cur_loss))
