import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    # TODO: Check about partition_weights, see original code
    # what i didn't understand is that for dice loss, partition_weights gets
    # multiplied inside the forward and also in the factory_loss function
    # I think that this is wrong, and removed it from the forward
    def __init__(self, classes):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes

    def forward(self, pred, gt):
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt)
        dice_score = (2 * intersection + self.eps) / (union + self.eps)
        return 1. - dice_score

