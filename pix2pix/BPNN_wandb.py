#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 10:53:32 2021

@author: rehan
"""

import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
from torchvision import transforms, utils
from torchvision.utils import make_grid
from natsort import natsorted
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import random
import pickle
import wandb

# GPU or CPU
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/media/rehan/Seagate Expansion Drive/sainbiose/New_datasets/HR_TIF_NOAUGMENTATION/Labels_norm_aug.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/media/rehan/Seagate Expansion Drive/sainbiose/New_datasets/HR_PNG_AUGMENTATION", help = "path to image directory")
parser.add_argument("--batch_size", default = 16, help = "number of batch")
parser.add_argument("--nof", default = 16, help = "number of filter")
parser.add_argument("--lr",default = 0.001, help = "learning rate")
parser.add_argument("--nb_epochs", default = 5, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "/media/rehan/Seagate Expansion Drive/src/nodropout", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "Using", help = "Mode used : Training, Using or Testing")
parser.add_argument("--cross_val", default = False, help = "mode training")
parser.add_argument("--k_fold", default = 5, help = "number of splitting for k cross-validation")


opt = parser.parse_args()
NB_LABEL = 34
PERCENTAGE_TEST = 20
RESIZE_IMAGE = 512

''' Config '''

config = dict(
    epochs = opt.nb_epochs,
    kernels = opt.nof,
    batch_size = opt.batch_size,
    learning_rate = opt.lr)

''' Database Creation '''

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx,0])
        image = io.imread(img_name) # Loading Image
        base = np.zeros((RESIZE_IMAGE,RESIZE_IMAGE)) # We need a 512x512 image to be at an order 2n without upscaling^
        base[6:506,6:506]=image # enelever les chiffres pour des variables
        image = base # Now, image has 512x512 pixels with a zero border
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        labels = self.labels.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        sample = {'image': image, 'label': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Test_Datasets(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.listdir(self.image_dir)
        img_name = os.path.join(self.image_dir,image_name[idx])
        image = io.imread(img_name) # Loading Image
        base = np.zeros((RESIZE_IMAGE,RESIZE_IMAGE)) # We need a 512x512 image to be at an order 2n without upscaling^
        base[6:506,6:506]=image # enelever les chiffres pour des variables
        image = base # Now, image has 512x512 pixels with a zero border
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample

    
''' Network '''

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # initialize CNN layers
        self.conv1 = nn.Conv2d(1,opt.nof,kernel_size = 3,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(opt.nof,opt.nof*2, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(opt.nof*2,opt.nof*4, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # initialize NN layers
        self.fc1 = nn.Linear(64*64*64,240)
        self.fc2 = nn.Linear(240,120)
        self.fc3 = nn.Linear(120,NB_LABEL)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
''' Training function '''

def train(model, train_loader, optimizer, criterion, epoch, opt, steps_per_epochs=20):
    model.train()
    print("starting training")
    print("----------------")
    train_loss = 0.0
    train_total = 0
    running_loss = 0.0
    r2_s = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data['image'], data['label']
        # reshape
        inputs = inputs.reshape(inputs.size(0),1,512,512)
        labels = labels.reshape(labels.size(0),NB_LABEL)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward backward and optimization
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        # statistics
        train_loss += loss.item()
        running_loss += loss.item()
        train_total += labels.size(0)
        outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        labels, outputs = np.array(labels), np.array(outputs)
        labels, outputs = labels.reshape(34,opt.batch_size), outputs.reshape(34,opt.batch_size)
        r2 = r2_score(labels,outputs)
        r2_s += r2
        
        if i % opt.batch_size == opt.batch_size-1:
            print('[%d %5d], loss: %.3f' %
                  (epoch + 1, i+1, running_loss/opt.batch_size))
            running_loss = 0.0
    # displaying results
    r2_s = r2_s/i
    print('Epoch [{}], Loss: {}, R square: {}'.format(epoch+1, train_loss/train_total, r2_s), end='')
    wandb.log({'Train Loss': train_loss/train_total, 'Train R square': r2_s, 'Epoch': epoch+1})
    
    print('Finished Training')
    # saving trained model
    check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
    torch.save(model.state_dict(),os.path.join(opt.checkpoint_path,check_name))

''' Testing function '''

def test(model, test_loader, criterion, epoch, opt):
    model.eval()
    
    test_loss = 0
    test_total = 0
    r2_s = 0
    output = {}
    label = {}
    # Loading Checkpoint
    if opt.mode is "Test":
        model = Net()
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path,check_name)))
    # Testing
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data['image'],data['label']
            # reshape
            inputs = inputs.reshape(1,1,512,512)
            labels = labels.reshape(1,NB_LABEL)
            # loss
            outputs = model(inputs)
            test_loss += criterion(outputs,labels)
            test_total += labels.size(0)
            # statistics
            outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            labels, outputs = np.array(labels), np.array(outputs)
            labels, outputs = labels.reshape(34,1), outputs.reshape(34,1)
            r2 = r2_score(labels,outputs)
            r2_s += r2
            print('r2 : %.3f , MSE : %.3f' %
                  (r2,test_loss))
            output[i] = outputs
            label[i] = labels
        name_out = "./output" + str(epoch) + ".txt"
        name_lab = "./label" + str(epoch) + ".txt"

        with open(name_out,"wb") as f:
            pickle.dump(output,f)
        with open(name_lab,"wb") as f:
            pickle.dump(label,f)
    
    r2_s = r2_s/i
    print(' Test_loss: {}, Test_R_square: {}'.format(test_loss/test_total, r2_s))
    if opt.mode == "Train":
        wandb.log({'Test Loss': test_loss/test_total, 'Test R square': r2_s})

""" Function of usage """

def using_mode(model, test_loader,opt):
    model.eval()
    epoch = 4
    output = {}
    label = {}
    # Loading Checkpoint
    model = Net()
    check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
    model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path,check_name)))
    # Testing
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data['image']
            # reshape
            inputs = inputs.reshape(1,1,512,512)
            # loss
            outputs = model(inputs)
            # statistics
            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(34,1)
            
        name_out = "./output" + str(epoch) + ".txt"

        with open(name_out,"wb") as f:
            pickle.dump(output,f)

    
''' main '''

# defining data
if opt.mode == "Train" or opt.mode == "Test":
    datasets = Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir) # Create dataset
else:
    datasets = Datasets(image_dir = opt.image_dir)
# defining the model
model = Net()

# defining the optimizer
optimizer =  Adam(model.parameters(), lr=opt.lr)
criterion = MSELoss()

# DATA SPLITING - Two possibilities : split data randomly or split data with specific index 
if opt.mode == "Train" or opt.mode == "Test":
    nb_data = len(datasets)
    nb_test = nb_data*(PERCENTAGE_TEST/100)
    nb_dataset = nb_data - nb_test
    # train,test = torch.utils.data.random_split(datasets, [int(nb_dataset),int(nb_test)]) # split data randomly
    ids = np.array(range(0,nb_data))
    random.shuffle(ids)
    train_ids = ids[0:int(nb_dataset)]
    test_ids = ids [int(nb_dataset):]
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = train_ids,  num_workers = 0 ) # Create batches and tensors
    testloader = DataLoader(datasets, batch_size = 1, sampler = test_ids, num_workers = 0 ) # Create batches and tensors
    # trainloader = DataLoader(train, batch_size = opt.batch_size, shuffle = True, num_workers = 0 ) # Create batches and tensors
    # testloader = DataLoader(test, batch_size = 1, shuffle = False, num_workers = 0 ) # Create batches and tensors
else:
    testloader = DataLoader(datasets,batch_size = 1, num_workers =0)
# training 

if opt.mode == "Train" or opt.mode == "Test":
    wandb.init(entity='jhuboo', project='BPNN1')
    wandb.watch(model, log='all')
    
    for epoch in range(opt.nb_epochs):
        train(model, trainloader, optimizer, criterion, epoch, opt)
        test(model, testloader, criterion, epoch, opt)    
    wandb.finish()
else:
    test(model, testloader, criterion,4,opt)









   
