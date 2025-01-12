import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import pathlib
import builtins
import socket
import random
import time
import json
# import pdb

import numpy as np
import torch
import torchio as tio
import torch.distributed as dist
import torch.utils.data as data

from hashlib import shake_256
from munch import Munch, munchify, unmunchify
from torch import nn
from os import path
from torch.backends import cudnn
from torch.utils.data import DistributedSampler
import wandb

from experiments.ExperimentFactory import ExperimentFactory
from dataloader.AugFactory import AugFactory


# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5)  # output len: 2*5=10
    return h.upper()


def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", default="config.yaml",
                            help="the config file to be used to run the experiment", required=True)
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # start wandb
    wandb.init(
        project="segment_task",
        # entity="maxillo",
        config=unmunchify(config)
    )

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # check if preprocessing is set and file exists
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preproc = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preproc = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preproc_file:
            preproc = yaml.load(preproc_file, yaml.FullLoader)
    config.data_loader.preprocessing = AugFactory(preproc).get_transform()

    # check if augmentations is set and file exists
    logging.info(f'loading augmentations')
    if config.data_loader.augmentations is None:
        aug = []
    elif not os.path.exists(config.data_loader.augmentations):
        logging.warning(f'Augmentations file does not exist: {config.augmentations}')
        aug = []
    else:
        with open(config.data_loader.augmentations) as aug_file:
            aug = yaml.load(aug_file, yaml.FullLoader)
    config.data_loader.augmentations = AugFactory(aug).get_transform()

    # make title unique to avoid overriding
    config.title = f'{config.title}_{timehash()}'

    logging.info(f'Instantiation of the experiment')
    # pdb.set_trace()
    experiment = ExperimentFactory(config, args.debug).get()
    logging.info(f'experiment title: {experiment.config.title}')

    project_dir_title = os.path.join(experiment.config.project_dir, experiment.config.title)
    os.makedirs(project_dir_title, exist_ok=True)
    logging.info(f'project directory: {project_dir_title}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(project_dir_title, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(project_dir_title, 'config.yaml')
    shutil.copy(args.config, copy_config_path)

    if not os.path.exists(experiment.config.data_loader.dataset):
        logging.error("Dataset path does not exist: {}".format(experiment.config.data_loader.dataset))
        raise SystemExit

    # pre-calculate the checkpoints path
    checkpoints_path = path.join(project_dir_title, 'checkpoints')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if experiment.config.trainer.reload and not os.path.exists(experiment.config.trainer.checkpoint):
        logging.error(f'Checkpoint file does not exist: {experiment.config.trainer.checkpoint}')
        raise SystemExit

    if config.experiment.name == 'Regression':
        best = float('inf')
    elif config.experiment.name == 'Segmentation':
        best = float('-inf')

    best_val = {
        'value': best,
        'epoch': -1
    }
    best_test = {
        'value': best,
        'epoch': -1
    }

    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        assert experiment.epoch < config.trainer.epochs
        for epoch in range(experiment.epoch, config.trainer.epochs + 1):
            experiment.train()

            val_metrics = experiment.test(phase="Validation")
            for val_metric, value in val_metrics.items():
                logging.info(f'Epoch {epoch} Val {val_metric}: {value}')

            # TODO: check if experiment name corresponds to eval_metric in yaml otherwise raise an error somewhere idk
            if config.experiment.name == 'Regression':
                config.experiment.metric_of_interest = 'mse'
            elif config.experiment.name == 'Segmentation':
                config.experiment.metric_of_interest = 'iou'
            val_evaluation_metric = val_metrics[config.experiment.metric_of_interest]

            if val_evaluation_metric < 1e-05 and experiment.epoch > 15:
                logging.warning('WARNING: drop in performances detected.')

            if experiment.scheduler is not None:
                if experiment.optimizer.name == 'SGD' and experiment.scheduler.name == 'Plateau':
                    experiment.scheduler.step(val_evaluation_metric)
                else:
                    experiment.scheduler.step(epoch)

            #if epoch % 1 == 0:
            #    experiment.predict_series(phase='Validation', rnd=True)

            experiment.save('last.pth')

            if (config.experiment.name == 'Regression' and val_evaluation_metric < best_val['value']) or (config.experiment.name == 'Segmentation' and val_evaluation_metric > best_val['value']):
                best_val['value'] = val_evaluation_metric
                logging.info(f'New best: {val_evaluation_metric}')
                best_val['epoch'] = epoch
                experiment.save('best.pth')

            experiment.epoch += 1

        logging.info(f'''
                Best validation {config.experiment.metric_of_interest} found: {best_val['value']} at epoch: {best_val['epoch']}
                ''')

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        experiment.load()
        test_iou, test_dice = experiment.test(phase="Test")
        logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}')

    # Do the inference
    if config.trainer.do_inference:
        logging.info('Doing inference...')
        experiment.load()
        experiment.inference(os.path.join(config.data_loader.dataset, 'SPARSE'))
        # experiment.inference('/homes/llumetti/out')

    # Do the prediction
    if config.trainer.do_prediction:
        logging.info('Doing prediction...')
        experiment.predict(config.trainer.predict)
        # experiment.predict_series()
        # experiment.inference('/homes/llumetti/out')

# TODO: add a Final test metric
