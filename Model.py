#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import misc
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import imageio
import torchvision.transforms.functional as TF
import gc
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
import copy


# In[7]:


epochs = 3
batch_size = 3
lr = 0.003
TRAIN_DATA_PATH = 'C:/Users/Balabhadrapatruni/Desktop/Learn/562468_1022626_compressed_Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
TEST_DATA_PATH = 'C:/Users/Balabhadrapatruni/Desktop/Learn/562468_1022626_compressed_Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader = data.DataLoader(test_data, batch_size=3, shuffle=True)
train_classes = train_data.classes
test_classes = test_data.classes
print(train_classes)
print(test_classes)


# In[8]:



def imshow(img, title):
    npimg = img.numpy() / 2 + 0.5
    plt.figure(figsize=(30, 3))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


# In[9]:


def show_batch_images(dataloader):
    images, labels = next(iter(dataloader))
    img = torchvision.utils.make_grid(images)
    print(labels)
    imshow(img, title=[str(train_classes[x.item()]) for x in labels])
    
show_batch_images(train_data_loader)


# In[10]:


resnet = models.resnet18()
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 7)
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(resnet.parameters(), lr=0.01)


# In[11]:


def evaluation(dataloader, model):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total


# In[7]:


loss_epoch_arr = []

min_loss = 1000

best_model = None
n_iters = np.ceil(5000/batch_size)

for epoch in range(epochs):

    for i, data in enumerate(train_data_loader, 0):

        inputs, labels = data
        opt.zero_grad()

        outputs = resnet(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
        if min_loss > loss.item():
            min_loss = loss.item()
            best_model = copy.deepcopy(resnet.state_dict())
            print('Min loss %0.2f' % min_loss)
        
        if i % 100 == 0:
            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))
            
        del inputs, labels, outputs
        torch.cuda.empty_cache()
        
    loss_epoch_arr.append(loss.item())
        
    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (
        epoch, epochs, 
        evaluation(test_data_loader, resnet), evaluation(train_data_loader, resnet)))
    
    
plt.plot(loss_epoch_arr)
plt.show()


# In[ ]:





# In[10]:


import pickle
resnet.load_state_dict(best_model)
filename = 'finalmodel.sav'
path = 'C:/Users/Balabhadrapatruni/Desktop/Learn/562468_1022626_compressed_Coronahack-Chest-XRay-Dataset/' + filename
pickle.dump(resnet, open(path, 'wb'))


# In[ ]:


print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (
        epoch, epochs, 
        evaluation(test_data_loader, resnet), evaluation(train_data_loader, resnet)))


# In[2]:


import pickle
filename = 'finalmodel.sav'
path = 'C:/Users/Balabhadrapatruni/Desktop/Learn/562468_1022626_compressed_Coronahack-Chest-XRay-Dataset/' + filename
load_model = pickle.load(open(path, 'rb'))


# In[3]:


print(load_model)


# In[14]:


print(' Test acc: %0.2f, Train acc: %0.2f' % (
        evaluation(test_data_loader, load_model), evaluation(train_data_loader, load_model)))


# In[ ]:




