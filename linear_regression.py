import numpy as np
import matplotlib.pyplot as plt

train_data = np.random.rand(1000, 2)
train_label = np.random.randint(0, 2, size=(1000, 1))
test_data = np.random.rand(1000, 2)
test_label = np.random.randint(0, 2, size=(1000, 1))

initial_weight = np.array([0.1, 0.2, 0.3])

def add_one_to_data(data):
    return np.insert(data, 0, 1, axis=1)

def loss_function(data, label, weight):
    loss = 0
    for i in range(len(data)):
        loss += (label[i] - np.dot(weight, data[i])) ** 2
    return loss/len(data)

def gradient_function(grad ,threshold):
    """比较梯度模长与阈值的大小，返回是否达到阈值"""
    if np.sqrt(np.dot(grad, grad)) < threshold:
        return True
    else:
        return False


def linear_regression(data, label,learning_rate, weight):
    loss = []
    print("data:", data.shape)
    print("label:", label.shape)
    print("weight:", weight.shape)
    while True:
        loss.append(loss_function(data, label, weight))
        print("loss:", loss[-1]) # 打印loss
        for i in range(len(data)):
            gradient = (label[i] - np.dot(weight, data[i])) * data[i]
            weight += learning_rate * gradient
            # print("gradient:", gradient)
        if len(loss) > 2 and abs(loss[-1] - loss[-2]) < 0.000001:
            break
    return weight,loss

def visualize_loss(loss):
    plt.figure()
    x = np.arange(0, len(loss))
    plt.plot(x, loss)
    plt.show()




train_data = add_one_to_data(train_data)
test_data = add_one_to_data(test_data)
weight,loss= linear_regression(train_data, train_label, 0.000001, initial_weight)
print("result weight:", weight)
visualize_loss(loss)
# print("train loss:", loss)