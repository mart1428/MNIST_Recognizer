import torch
import torch.nn 
import torch.nn.functional as F

import torchvision
import torchvision.datasets
import torchvision.transforms

import torch.optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_relevant_indices(dataset, classes, target_classes):

    #classes = dog, cat, horse
    # target classes = dog, cat 


    indices = []
    for i in range(len(dataset)):
        label_index = dataset[i][1]
        label_class = classes[label_index]

        if label_class in target_classes:
            indices.append(i)
    
    return indices

def get_data_loader(trainval_data, test_data, classes, target_classes, batch_size= 128):
    
    trainval_data_indices = get_relevant_indices(trainval_data, classes, target_classes)
    test_data_indices = get_relevant_indices(test_data, classes, target_classes)

    split = int(len(trainval_data_indices) * 0.8)
    train_indices = trainval_data_indices[:split]
    val_indices = trainval_data_indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_data_indices)

    train_loader = DataLoader(trainval_data, batch_size, sampler = train_sampler)
    val_loader = DataLoader(trainval_data, batch_size, sampler = val_sampler)

    test_loader = DataLoader(test_data, batch_size, sampler = test_sampler)

    return train_loader, val_loader, test_loader
