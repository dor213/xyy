import numpy as np
import matplotlib.pyplot as plt


def prepare_data():
    np.random.seed(1)
    x1 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, -5], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

def add_one_to_data(data):
    return np.insert(data, 0, 1, axis=1)

def data_remove_one(data):
    data = data[:, 1:]
    return data

def loss_function(data, label, weight, C):
    loss = 0
    for i in range(len(data)):
        loss += max(0, 1 - label[i]*np.dot(weight, data[i]))
    return loss/len(data) + C*np.dot(weight, weight)

def dual_loss_function(data, label, alpha, C):
    loss = 0
    for i in range(len(data)):
        for j in range(len(data)):
            loss += alpha[i]*alpha[j]*label[i]*label[j]*np.dot(data[i], data[j])
    loss = loss/len(data) - np.sum(alpha)
    return loss

def prime_svm(data, label, C):
    w = np.zeros(len(data[0]))
    loss = []
    for i in range(1000):
        loss.append(loss_function(data, label, w, C))
        for j in range(len(data)):
            if label[j]*np.dot(w, data[j]) < 1:
                w = w + 0.01*(data[j]*label[j] - 2*C*w)
            else:
                w = w - 0.01*2*C*w
    return w, loss

def dual_svm(data, label, C):
    alpha = np.zeros(len(data))
    loss = []
    for i in range(1000):
        loss.append(dual_loss_function(data, label, alpha, C))
        for j in range(len(data)):
            alpha[j] = alpha[j] + 0.01*(1 - label[j]*np.dot(alpha*label, np.dot(data, data[j])))
            if alpha[j] < 0:
                alpha[j] = 0
            elif alpha[j] > C:
                alpha[j] = C
    w = np.dot(alpha*label, data)
    return w, loss
