# -*- coding: utf-8 -*-
"""PyTorch dataloader"""

from torchvision import datasets
from torch.utils.data import DataLoader
from configs.config import CFG
def create_datasets(**params):
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)

    return train_data, test_data

def create_dataloaders(**params):
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1, # how many samples per batch?
                                  num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False) # don't usually need to shuffle testing data

    return train_dataloader, test_dataloader


