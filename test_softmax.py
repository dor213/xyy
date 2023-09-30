import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import IPython as ipy
import cv2

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=2)



W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)

W1 = torch.normal(0, 0.01, size=(784, 128), requires_grad=True)
b1 = torch.zeros(128, requires_grad=True)
W2 = torch.normal(0, 0.01, size=(128, 10), requires_grad=True)
b2 = torch.zeros(10, requires_grad=True)

def Softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)


def model(x):
    x = x.reshape(-1, 784)
    y = torch.matmul(x, W) + b 
    return Softmax(y)

def model_2(x):
    x = x.reshape(-1, 784)
    y = torch.matmul(x, W1) + b1 
    y = torch.relu(y)
    y = torch.matmul(y, W2) + b2 
    return Softmax(y)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(y.shape[0]), y]).mean()

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(torch.sum(cmp.type(y.dtype)))

def evaluate_accuracy(data_loader, net):
    acc_sum, n = 0.0, 0
    for x, y in data_loader:
        pred = net(x)
        acc_sum += accuracy(pred, y)
        n += 1
    return acc_sum / n

def train(net, train_loader, test_loader, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            y_hat = net(x)
            l = loss(y_hat, y).sum()
            l.requires_grad_(True)  
            updater.zero_grad()
            l.backward()
            updater.step()

            train_loss_sum += l
            train_acc_sum += accuracy(y_hat, y)
            n += x.shape[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc/100))
    return net

def test(net, test_loader):
    # net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(images.shape[0]):
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title(f"Predicted: {predicted[i]}, Actual: {labels[i]}")
                plt.show()

if __name__ == '__main__':
    num_epochs, lr = 5, 0.1
    updater = torch.optim.SGD([W1, b1, W2, b2], lr=lr,momentum=0.9)
    '''训练model_2'''
    print('训练model_2')
    result = train(model_2, train_loader, test_loader, cross_entropy, num_epochs, updater)
    '''测试model_2'''
    print('测试model_2')
    test(result, test_loader)



