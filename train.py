import torch
import logging
from tqdm import tqdm
from torch import nn


def train(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, warmup):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels, warmup)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # final predictions
        if outputs.shape[1] > 1:
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        else:
            outputs = nn.Sigmoid()(outputs)
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0
            outputs = outputs.squeeze().cpu().detach().numpy()

        labels = labels.cpu().numpy()
        evaluator.iou(outputs, labels)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_train_metric = evaluator.mean_metric()
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Metric/train', epoch_train_metric, epoch)

    logging.info(
        f'Train Epoch [{epoch}], '
        f'Train Mean Loss: {epoch_train_loss}, '
        f'Train Mean Metric: {epoch_train_metric}'
    )

    return epoch_train_loss, epoch_train_metric