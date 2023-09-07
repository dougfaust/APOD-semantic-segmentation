# -*- coding: utf-8 -*-
"""PyTorch dataloader"""

from torchvision import datasets
from torch.utils.data import DataLoader
from configs.config import CFG

class DataLoader:

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
        train_data = datasets.ImageFolder(root=CFG['data']['train_dir'], # target folder of images
                                          transform=CFG['data']['data_transform'], # transforms to perform on data (images)
                                          target_transform=None) # transforms to perform on labels (if necessary)

        test_data = datasets.ImageFolder(root=CFG['data']['test_dir'],
                                         transform=CFG['data']['data_transform'])

        return train_data, test_data

    @staticmethod
    def create_dataloaders():

        train_data, test_data = create_datasets()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=1, # how many samples per batch?
                                      num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                      shuffle=True) # shuffle the data?

        test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False)

        return train_dataloader, test_dataloader


