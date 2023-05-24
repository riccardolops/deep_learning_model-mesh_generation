import json
import logging
import logging.config
from pathlib import Path
import os

import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader

import nibabel as nib

class Heart(tio.SubjectsDataset):
    """
    Heart dataset
    TODO: Add more information about the dataset
    """

    def __init__(self, root, splits, transform=None, dist_map=None, **kwargs):
        if type(dist_map) == str:
            dist_map = [dist_map]

        root = Path(root)
        if not isinstance(splits, list):
            splits = [splits]

        subjects_list = self._get_subjects_list(root, splits, dist_map)
        super().__init__(subjects_list, transform, **kwargs)

    def _get_subjects_list(self, root, splits, dist_map=None):

        with open(os.path.join(root, 'dataset_new.json')) as dataset_file:
            json_dataset = json.load(dataset_file)

        if dist_map is None:
            dist_map = []

        subjects = []
        for split in splits:
            if split=='train':
                dataset = json_dataset[:10]
                for patient in dataset:
                    # TODO: add naive volume
                    subject_dict = {
                        'partition': split,
                        'patient': patient,
                        'data': tio.ScalarImage(root / patient['image']),
                        'dense': tio.LabelMap(root / patient['label']),
                        'distance': tio.ScalarImage(root / patient['distance']),
                    }
                    subjects.append(tio.Subject(**subject_dict))
                print(f"Loaded {len(subjects)} patients for split {split}")
            elif split == 'val':
                dataset = json_dataset[-10:]
                for patient in dataset:
                    # TODO: add naive volume
                    subject_dict = {
                        'partition': split,
                        'patient': patient,
                        'data': tio.ScalarImage(root / patient['image']),
                        'dense': tio.LabelMap(root / patient['label']),
                        'distance': tio.ScalarImage(root / patient['distance']),
                    }
                    subjects.append(tio.Subject(**subject_dict))
                print(f"Loaded {len(subjects)} patients for split {split}")
            elif split=='test':
                dataset = json_dataset[-10:]
                for patient in dataset:
                    # TODO: add naive volume
                    subject_dict = {
                        'partition': split,
                        'patient': patient,
                        'data': tio.ScalarImage(root / patient['image']),
                        'dense': tio.LabelMap(root / patient['label']),
                        'distance': tio.ScalarImage(root / patient['distance']),
                    }
                    subjects.append(tio.Subject(**subject_dict))
                print(f"Loaded {len(subjects)} patients for split {split}")
            else:
                logging.error("Dataset '{}' does not exist".format(split))
                raise SystemExit
        return subjects

    def get_loader(self, config, aggr=None):
        samples_per_volume = [np.round(i / (j - config.grid_overlap)) for i, j in zip(config.resize_shape,
                                                                                      config.patch_shape)]
        samples_per_volume = int(np.prod(samples_per_volume))
        print(f"There are {samples_per_volume} samples per volume")
        # sampler = tio.GridSampler(patch_size=config.patch_shape, patch_overlap=config.grid_overlap)
        sampler = tio.UniformSampler(patch_size=config.patch_shape)
        queue = tio.Queue(
            subjects_dataset=self,
            max_length=100,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=config.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
            start_background=True,
        )
        loader = DataLoader(queue, batch_size=config.batch_size, num_workers=0, pin_memory=True)
        return loader

    def get_aggregator(self, config, aggr=None):
        samplers = [tio.GridSampler(sj, patch_size=config.patch_shape, patch_overlap=0) for sj in self._subjects]
        return [(test_p, DataLoader(test_p, 2, num_workers=4)) for test_p in samplers]

    # def get_aggregator(self, config, aggr=None):
    #     for subject in self._subjects:
    #         sampler = tio.GridSampler(subject, patch_size=config.path_shape, patch_overlap=0)
