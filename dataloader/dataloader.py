# -*- coding: utf-8 -*-
"""PyTorch dataloader"""

import os
from torchvision import datasets
from torch.utils.data import DataLoader
from configs.config import CFG

class ClassifierDataLoader:

    def __init__(self):
        """
        :param CFG:
        """

        self.config = CFG

    @staticmethod
    def return_config():
        return CFG

    @staticmethod
    def create_datasets():
        train_data = datasets.ImageFolder(root=CFG['data']['train_dir'],
                                          transform=CFG['data']['data_transform'],
                                          target_transform=None) # transforms to perform on labels

        test_data = datasets.ImageFolder(root=CFG['data']['test_dir'],
                                         transform=CFG['data']['data_transform'])

        return train_data, test_data

    @staticmethod
    def create_dataloaders():

        train_data, test_data = ClassifierDataLoader.create_datasets()

        num_workers = CFG['train']['num_workers'] if CFG['train']['num_workers'] is not None else os.cpu_count()
        num_workers = min(num_workers, os.cpu_count()) # check to ensure config param is compatible with hardware

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=CFG['train']['batch_size'],
                                      num_workers=num_workers,
                                      shuffle=True)

        test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=CFG['train']['batch_size'],
                                     num_workers=num_workers,
                                     shuffle=False)

        return train_dataloader, test_dataloader


