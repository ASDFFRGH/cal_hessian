import torch
import torchvision
from torch import nn
import torch.optim as optim
from model import MFNN1, MFNN2, MLP
from utils import *
import numpy as np
#import pandas as pd
import sys
import time
from tqdm import tqdm
from dataclasses import dataclass
from sam import SAM
from nsam import NSAM

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
config = configs(model='MFNN1', noise_type='NONE', scheduler=1, dataset='FashionMNIST', lr = 0.1, bs=128, rho=10)

root = make_root(config)
path = get_path(root)

loader_train, loader_test = make_data_loader(config)

print(f'printed logs in {path}')

def train_sam(model, opt, loss_fn, X, t):
    X, t = X.to(device), t.to(device)
    y = model(X)
    loss = loss_fn(y, t)
    opt.zero_grad() 
    loss.backward()
    opt.first_step(zero_grad=True)
    loss_fn(model(X), t).backward()
    opt.second_step(zero_grad=True)

    pred = y.argmax(1) 
    
    return loss, pred

def train_sgd(model, opt, loss_fn, X, t):
    X, t = X.to(device), t.to(device)
    y = model(X)
    loss = loss_fn(y, t)
    opt.zero_grad() 
    loss.backward()
    opt.step()
    
    pred = y.argmax(1) 
    
    return loss, pred

def test(model, loss_fn, X, t):
    X, t = X.to(device), t.to(device)
    y = model(X)
    loss = loss_fn(y, t)

    pred = y.argmax(1)
    
    return loss, pred


def main(model, opt, loss_fn, scheduler, config):
    with open(path, 'a') as f:
        f.write(f'{config}')
        f.write('\n')
    print(config.noise_type)
    rho = config.rho
    for epoch in range(config.epochs):
        with open(path, 'a') as f:
            f.write(f'EPOCH: {epoch}\n')
        print(f'EPOCH: {epoch}')
        
        start = time.time() 
        
        #start train loop----------------------
        train_loss = []
        total_train = 0
        correct_train = 0
        
        model.train()
        for X, t in tqdm(loader_train):
            X, t = X.to(device), t.to(device)
            if config.noise_type == 'NONE':
                loss, pred = train_sgd(model, opt, loss_fn, X, t)
            elif config.noise_type == 'FINITE':
                loss, pred = train_sam(model, opt, loss_fn, X, t)
            elif config.noise_type == 'SAM':
                loss, pred = train_sam(model, opt, loss_fn, X, t)
                
            train_loss.append(loss.tolist())
            
            total_train += t.shape[0]
            correct_train += (pred==t).sum().item()
        scheduler.step()
    
        end  = time.time()
        
        log = f'train loss: {np.mean(train_loss):.3f}, accuracy: {correct_train/total_train:.3f}'
        with open(path, 'a') as f:
            f.write(log + f' train_time: {end - start:.5f}' + '\n')
        print(log)
        #end train loop------------------------------------
        
                
        #start test loop-----------------------------------
        test_loss = []
        total_test= 0
        correct_test = 0
    
        model.eval()
        for X, t in loader_test:
            loss, pred = test(model, loss_fn, X, t)
            test_loss.append(loss.tolist())
            
            total_test += t.shape[0]
            correct_test += (pred==t).sum().item()
    
        log = f'test loss: {np.mean(test_loss):.3f}, accuracy: {correct_test/total_test:.3f}'
        with open(path, 'a') as f:
            f.write(log + '\n')
        print(log)
        # end test loop-------------------------------------- 
        
        print(f'time: {end - start:.5f}')           

def set_up_opt(config, model):

    if config.noise_type == 'FINITE':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        opt = NSAM(model.parameters(), base_optimizer, lr=config.lr, momentum=config.momentum, rho=config.rho)
    elif config.noise_type == 'NONE':
        opt = optim.SGD(model.parameters(), lr = config.lr, weight_decay = config.weight_decay, momentum = config.momentum)
    elif config.noise_type == 'SAM':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        opt = NSAM(model.parameters(), base_optimizer, lr=config.lr, momentum=config.momentum, rho=config.rho)
    
    return opt

def set_up_model(config):  
    if config.model == 'MFNN1':
        model = MFNN1()
    return model

def set_up_scheduler(config, opt): 
    if config.scheduler == 1:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda = func, verbose=True)
        
    return scheduler

model= set_up_model(config)
opt = set_up_opt(config, model)
scheduler = set_up_scheduler(config, opt)
loss_fn = nn.CrossEntropyLoss()

main(model, opt, loss_fn, scheduler, config)






     