import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def sign(x):
    if x > 0:
        return 1
    else:
        return -1
    
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

def svm(data, label, learning_rate, epochs, init_w, batch_size, C):
    w = init_w
    loss = []
    for i in range(epochs):
        # print("epoch:", i)
        loss.append(loss_function(data, label, w, C))
        for j in range(0, data.shape[0], batch_size):
            gradient = np.zeros(len(w))
            for k in range(batch_size):
                if label[j+k]*np.dot(w, data[j+k]) < 1:
                    gradient = gradient + data[j+k]*label[j+k]
            gradient = gradient/ batch_size
            w = w - learning_rate*gradient
    return w,loss




def svm_qp(data, label, C):
    P = matrix(np.dot(data.T, data))
    q = matrix(-np.ones(len(data)))
    G = matrix(np.vstack((-np.eye(len(data)), np.eye(len(data)))))
    h = matrix(np.hstack((np.zeros(len(data)), np.ones(len(data))*C)))
    A = matrix(label.reshape(1, -1))
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x']).reshape(-1)
    return w
    



def visualize_data_and_result(data,label, w):
    plt.scatter(data[:, 1], data[:, 2], c=label)
    x = np.arange(-5, 5, 0.1)
    y = -(w[0] + w[1] * x) / w[2]
    plt.plot(x, y)
    plt.show()

def main():
    x1, label1, x2, label2 = prepare_data()
    train_data = np.concatenate((x1, x2), axis=0)
    train_label = np.concatenate((label1, label2), axis=0)
    train_data = add_one_to_data(train_data)
    train_label = train_label.reshape(-1, 1)
    init_w = np.zeros(train_data.shape[1])
    w,loss = svm(train_data, train_label, 0.01, 100, init_w, 10, 0.1)
    w_qp = svm_qp(train_data, train_label, 0.1)
    print("w:", w)
    print("w_qp:", w_qp)
    # visualize_data_and_result(train_data, train_label, w)
    # plt.plot(loss)
    # plt.show()

if __name__ == '__main__':
    main()