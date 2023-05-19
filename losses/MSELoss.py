import torch


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.shape == gt.shape, 'prediction and ground truth shapes must match'
        # Calculate the mean squared error (MSE) loss
        loss = torch.mean((pred - gt) ** 2)
        return loss
