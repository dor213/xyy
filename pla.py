import numpy as np
import matplotlib.pyplot as plt

inital_pla_weight = np.array([10.5, -0.32, 0.10])  # 初始化pla算法的权重
inital_pocket_weight = np.array([0.05, 0.106, 10.07])  # 初始化pocket算法的权重

def prepare_data():
    np.random.seed(1)
    x1 = np.random.multivariate_normal([-5, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

def data_add_one(data):
    data = np.hstack((np.ones((len(data), 1)), data))
    return data

def data_remove_one(data):
    data = data[:, 1:]
    return data

def sign(x):
    return 1 if x >= 0 else -1

def pla(train_data, label, weight,num):
    """pla算法"""
    for i in range(num):
        for j in range(len(train_data)):
            if sign(np.dot(train_data[j], weight)) != label[j]:
                weight += label[j] * train_data[j]
                # break
        # print("pla i:", i, "w:", weight)
    print("pla final result", weight)
    print("pla iterative num", num)
    return weight


def pocket(train_data, label, weight,pocket_weight, num):
    """pocket算法"""
    w_error = 0
    pocket_error = 0
    for i in range(num):
        for j in range(len(train_data)):
            if sign(np.dot(train_data[j], weight)) != label[j]:
                weight += label[j] * train_data[j]
                for k in range(len(train_data)):
                    if sign(np.dot(train_data[k], weight)) != label[k]:
                        w_error += 1
                    if sign(np.dot(train_data[k], pocket_weight)) != label[k]:
                        pocket_error += 1
                if w_error < pocket_error:
                    pocket_weight = weight.copy()
    print("pocket final result", pocket_weight)
    print("pocket iterative num", num)
    return pocket_weight

def visualize_data_and_classfication_surface(data, label, weight):
    """可视化数据和分类面,label为1用圆圈表示，label为-1用叉表示，weight为权重,weight[0]+weight[1]*x+weight[2]*y=0,直线在x轴上的投影范围为[0,6]"""
    plt.figure()
    plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], marker='o', color='r', label='label=1')
    plt.scatter(data[label == -1][:, 0], data[label == -1][:, 1], marker='x', color='b', label='label=-1')
    x=np.linspace(-10,10,100)
    y=(-weight[0]-weight[1]*x)/weight[2]
    plt.plot(x,y,label='classification surface')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x1, label1, x2, label2 = prepare_data()
    x1 = data_add_one(x1)
    x2 = data_add_one(x2)
    train_data = np.vstack((x1[:160], x2[:160]))
    train_label = np.hstack((label1[:160], label2[:160]))
    test_data = np.vstack((x1[160:], x2[160:]))
    test_label = np.hstack((label1[160:], label2[160:]))
    pla_weight = pla(train_data, train_label, inital_pla_weight,100)
    pocket_weight = pocket(train_data, train_label, inital_pocket_weight, inital_pocket_weight,100)
    pla_correct = 0
    pocket_correct = 0
    for i in range(len(test_data)):
        if sign(np.dot(test_data[i], pla_weight)) == test_label[i]:
            pla_correct += 1
        if sign(np.dot(test_data[i], pocket_weight)) == test_label[i]:
            pocket_correct += 1
    test_data = data_remove_one(test_data)
    print("pla correct rate", pla_correct / len(test_data))
    print("pocket correct rate", pocket_correct / len(test_data))
    visualize_data_and_classfication_surface(test_data, test_label, pocket_weight)



    