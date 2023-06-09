import torch
import torch.nn as nn

class L1_LproductLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-06

    def forward(self, pred, gt):
        l1_loss = nn.L1Loss()
        loss = -torch.sum((gt * pred) / (gt * pred + pred**2 + gt**2 + self.eps))
        return l1_loss(pred, gt) + loss
