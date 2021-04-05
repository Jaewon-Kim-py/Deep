#import library
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

#hyper parameters
batch_size = 32
epochs = 10
learning_rate = 0.01

#load dataset
train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",train = True, download =True, transform = transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",train = False, download =True, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle =True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle =False)

#build model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

#define model
model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train and valid
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss = []
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward and optimize
        
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
    .format(epoch+1, num_epochs, i+1, total_step, sum(total_loss)/len(total_loss)))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        val_acc = 100 * correct / total
        print('Test Accuracy of the model on the test images: {} %'.format(val_acc))
