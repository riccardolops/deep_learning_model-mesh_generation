import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes
        self.weights = partition_weights

    def forward(self, pred, gt):
        included = [v for k, v in self.classes.items() if k not in ['UNLABELED']]
        gt_onehot = torch.nn.functional.one_hot(gt.squeeze().long(), num_classes=len(self.classes))
        if gt.shape[0] == 1:  # we need to add a further axis after the previous squeeze()
            gt_onehot = gt_onehot.unsqueeze(0)

        gt_onehot = torch.movedim(gt_onehot, -1, 1)
        input_soft = F.softmax(pred, dim=1)
        dims = (2, 3, 4)

        intersection = torch.sum(input_soft * gt_onehot, dims)
        cardinality = torch.sum(input_soft + gt_onehot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return 1. - dice_score[:, included]

