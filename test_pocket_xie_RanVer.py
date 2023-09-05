import numpy as np
import random
#hahaha
train_data = np.array([[1,0.2,0.7]
                       ,[1,0.3,0.3]
                       ,[1,0.4,0.5]
                       ,[1,0.6,0.5]
                       ,[1,0.1,0.4]
                       ,[1,0.4,0.6]
                       ,[1,0.6,0.2]
                       ,[1,0.7,0.4]
                       ,[1,0.8,0.6]
                       ,[1,0.7,0.5]])

train_label = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
w0 = [1,1,1] # initial w
pocket_w = [1,1,1]

def sign(x):
    if x >= 0 :
        return 1
    else:
        return -1
    

def train_with_pocket(train_data,train_label,w0,pocket_w):
    w = w0
    w_error = 0
    pocket_error = 0
   #ErrDot = []
    DotSeq = []
    for p in range(len(train_data)):
        DotSeq.append(p)
    RanDot = DotSeq.copy()
    random.seed()
    
    for i in range(20):
        #RanDot = DotSeq.copy()
        for j  in range(len(train_data)):
            #w_error = 0
            #pocket_error = 0
            m = random.choice(RanDot)
            RanDot.remove(m)
            if len(RanDot) == 0:
                RanDot = DotSeq.copy()
            #print("RanSeq:",RanDot,m)
            #print("DotSeq:",DotSeq)
            if sign(np.dot(w,train_data[m])) != train_label[m]:
                w = w + train_label[m]*train_data[m]
                break
        for k in range(len(train_data)):
            if sign(np.dot(w,train_data[k])) != train_label[k]:
                    w_error += 1
            if sign(np.dot(pocket_w,train_data[k])) != train_label[k]:
                    pocket_error += 1
        if w_error < pocket_error:
                pocket_w = w
        print("i:",i,"w_Renew",w,"w_error:",w_error,"pocket_error:",pocket_error,"pocket_w:",pocket_w)
        w_error = 0
        pocket_error = 0
    return pocket_w

pocket_w = train_with_pocket(train_data,train_label,w0,pocket_w)    
print("result w:",pocket_w) 