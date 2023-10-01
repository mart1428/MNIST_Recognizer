import torch
import torch.nn 
import torch.nn.functional as F

import torchvision
import torchvision.datasets
import torchvision.transforms

import torch.optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data_loader import get_data_loader
from model import MNIST_model
from train import train_mnist

def retrieve_data():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5), (0.5))
    ])

    trainval_data = torchvision.datasets.MNIST(root='./train_data', train = True, transform = transform, download = True)
    test_data = torchvision.datasets.MNIST(root='./test_data', train = False, transform = transform, download = True)

    return trainval_data, test_data


if __name__ == '__main__':
    torch.manual_seed(1000)


    trainval_data, test_data = retrieve_data()
    classes = trainval_data.classes
    target_classes = classes
    
    batch_size = 128
    lr = 0.001
    epochs = 30
    train_loader, val_loader, test_loader = get_data_loader(trainval_data, test_data, classes, target_classes, batch_size)
    
    model = MNIST_model()
    train_mnist(model, train_loader, val_loader, batch_size, lr = lr, epochs = epochs)

