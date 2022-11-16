# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Wrapper to train/test models."""

import argparse
import sys

import mvit.utils.checkpoint as cu
from engine import test, train
from mvit.config.defaults import assert_and_infer_cfg, get_cfg
from mvit.utils.misc import launch_job

import pprint

import mvit.models.losses as losses
import mvit.models.optimizer as optim
import mvit.utils.checkpoint as cu
import mvit.utils.distributed as du
import mvit.utils.logging as logging
import mvit.utils.metrics as metrics
import mvit.utils.misc as misc
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import math
from mvit.datasets import loader
from mvit.datasets.mixup import MixUp
from mvit.models import build_model
from mvit.utils.meters import EpochTimer, TrainMeter, ValMeter

logger = logging.get_logger(__name__)

from main import parse_args, load_config

def save_checkpoint(epoch, model, optimizer, scaler, scheduler, path):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scaler, scheduler, path):
    checkpoint = torch.load(path)
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return epoch

def train(epoch, model, trainloader, device, scaler, criterion, optimizer):
  running_loss = 0.0
  model.train()
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    for k in range(len(data)):
        data[k] = data[k].to(device)
    inputs, labels = data

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    # print statistics
    # print(f"{i}: {loss.item()}")
    running_loss += loss.item()
    #print("loss:", i, loss.item())
    if i % 100 == 99:    # print every 20 mini-batches
        print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
        running_loss = 0.0  
    

def test(model, testloader, device, criterion, epoch=-1):
    model.eval()
    correct = 0
    total = 0
    cnt=0
    running_loss=0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            for k in range(len(data)):
                data[k] = data[k].to(device)
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            cnt+=1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if epoch == -1:
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.3f} %, loss: {running_loss/cnt:.3f}')
    else:
        print(f'Accuracy of the network on the 10000 test images for epoch {epoch}: {100 * correct / total:.3f} %, loss: {running_loss/cnt:.3f}')

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # print("START OF CFG")
    # print(cfg)
    # print("END OF CFG")

    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)
    #model state loaded
    print("MODEL LOADED SUCCESSFULLY") 

    #some hyperparameters
    batch_size = 64
    img_size = 224
    lr = 0.00001
    n_epochs=40

    #load in cifar 10
    #https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    gpu=0
    device=torch.device(f"cuda:{gpu}")
    # device=torch.device("cpu")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.1) paper said no weight decay for fine tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    load_model = False
    start_epoch = 0
    if load_model:
        #U P D A T E  T H I S
        start_epoch = load_checkpoint(model, optimizer, scaler, scheduler, "cifar10_models/MVITv2_B_Cifar10_224_0.pth") 
        start_epoch += 1

    for epoch in range(start_epoch, n_epochs):  # loop over the dataset multiple times
        train(epoch, model, trainloader, device, scaler, criterion, optimizer)

        test(model, testloader, device, criterion, epoch)
        
        scheduler.step()
        if epoch % 1 == 0:
            save_checkpoint(epoch, model, optimizer, scaler, scheduler, f"cifar10_models/MVITv2_B_Cifar10_{img_size}_{epoch}.pth")
            print(f"Saved checkpoint to cifar10_models/MVITv2_B_Cifar10_{img_size}_{epoch}.pth in google drive")
    print('Finished Training')

if __name__ == "__main__":
    main()
