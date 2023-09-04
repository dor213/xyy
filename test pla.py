import numpy as np
import random


train_data = np.array([[1,3,3],[1,4,3],[1,1,1]])
train_label = np.array([1,1,-1])
test_data = np.array([1,0,1])
w0 = [0,3,1] # initial w

def sign(x):
    if x > 0 or x == 0:
        return 1
    else:
        return -1
    
def train(train_data,train_label,w0):
    w = w0
    for i in range(10):
        for j  in range(len(train_data)):
            if sign(np.dot(w,train_data[j])) != train_label[j]:
                w = w + train_label[j]*train_data[j]
                break
        print("i:",i,"w:",w)
    return w

w = train(train_data,train_label,w0)
print("result w:",w)
print("test result:",sign(np.dot(w,test_data)))