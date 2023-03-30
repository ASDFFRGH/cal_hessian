import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader 
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from dataclasses import dataclass
import os

def make_root(configs):
    root = 'logs/log_' + configs.dataset + '_' + configs.opt_type
    return root

def get_path(root):
    i = 1
    while True:
        path = root + '_' + str(i)
        is_file = os.path.isfile(path)
        if is_file:
            i += 1
        else:
            break
        
    return path

def func(epoch):
    if epoch < 50:
        return 0.5**0
    elif epoch < 70:
        return 0.5**1
    elif epoch < 90:
        return 0.5**2
    elif epoch < 150:
        return 0.5**3
    
def func2(epoch):
    if epoch < 50:
        return 0.5**0
    elif epoch < 70:
        return 0.5**1
    else: return 0.5**1

    
    
def make_data_loader(config):
    
    if config.dataset == 'FashionMNIST':
        train = FashionMNIST('FashionMNIST', 
                      train = True, 
                      download = True, 
                      transform = transforms.ToTensor()
                      )
        test = FashionMNIST('FashionMNIST',
                       train = False,
                       download = True,
                       transform = transforms.ToTensor()
                       )
    else:
        print('detasetが不明です．')

    X_train = train.data.type(torch.float32)
    t_train = train.targets

    X_test = test.data.type(torch.float32)
    t_test = test.targets
    
    ds_train = TensorDataset(X_train, t_train) 
    ds_test = TensorDataset(X_test, t_test)
    
    loader_train = DataLoader(ds_train, batch_size = config.bs, shuffle = True)
    loader_test = DataLoader(ds_test, batch_size = 10000, shuffle = True)

    return loader_train, loader_test
    
@dataclass
class configs():
    model: str
    opt_type: str
    scheduler: int
    dataset: str
    lr: float
    bs: int
    weight_decay: float = 0.0
    momentum: float = 0.0
    rho: int = 1
    epochs: int = 10
        