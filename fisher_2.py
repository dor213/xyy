import numpy as np
import matplotlib.pyplot as plt


def prepare_data():
    np.random.seed(1)
    x1 = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], 200)
    label1 = np.ones(len(x1))
    x2 = np.random.multivariate_normal([0, 1], [[1, 0], [0, 1]], 200)
    label2 = np.ones(len(x2)) * -1
    return x1, label1, x2, label2

def add_one_to_data(data):
    return np.insert(data, 0, 1, axis=1)


def data_remove_one(data):
    data = data[:, 1:]
    return data


def calculate_average(data):
    r1 = 0
    r2 = 0
    result_array = np.zeros(2)
    for i in range(len(data)):
        r1 = r1+ data[i][0]
        r2 = r2+data[i][1]
    r1 = r1/len(data)
    r2 =r2/len(data)
    result_array[0]=r1
    result_array[1]=r2
    return result_array

def calculate_covariance(data):
    """计算数据的协方差"""
    data_mean = calculate_average(data)
    print("data_mean",data_mean)
    data_covariance = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(len(data)):
        data_covariance[0][0] = data_covariance[0][0] + (data[i][0]-data_mean[0])**2
        data_covariance[0][1] = data_covariance[0][1] + (data[i][0]-data_mean[0])*(data[i][1]-data_mean[1])
        data_covariance[1][0] = data_covariance[1][0] + (data[i][1]-data_mean[1])*(data[i][0]-data_mean[0])
        data_covariance[1][1] = data_covariance[1][1] + (data[i][1]-data_mean[1])**2
    # print("data_covariance",data_covariance)
    return data_covariance

def fisher(data1,data2):
    """计算fisher线性判别函数的权重"""
    data1_covariance = calculate_covariance(data1)
    data2_covariance = calculate_covariance(data2)
    print("data1_covariance",data1_covariance)
    print("data2_covariance",data2_covariance)
    data1_mean = calculate_average(data1)
    data2_mean = calculate_average(data2)
    print("data1_mean",data1_mean)
    print("data2_mean",data2_mean)
    sw= data1_covariance + data2_covariance
    print("sw",sw)
    sw_inverse = np.linalg.inv(sw)
    w_best = np.dot(sw_inverse,data1_mean-data2_mean)
    print("w_best",w_best)
    w_best_T = w_best.T
    w_judge_threshold = np.dot(w_best_T,data1_mean+data2_mean)/2
    print("w_judge_threshold",w_judge_threshold)
    return w_best,w_judge_threshold


def visualize_data_and_fisher_line(data1,data2,w_best,w_judge_threshold):
    """可视化数据和fisher线性判别函数的分类线"""
    plt.plot(data1[:, 0], data1[:, 1], 'bo', label='data1')
    plt.plot(data2[:, 0], data2[:, 1], 'ro', label='data2')
    x = np.linspace(-10, 10, 100)
    y = (-w_best[0] * x - w_judge_threshold) / w_best[1]
    plt.plot(x, y, label='fisher line')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data1,label1,data2,label2 = prepare_data()
    train_data1 = data1[:160]
    train_data2 = data2[:160]
    test_data1 = data1[160:]
    test_data2 = data2[160:]
    w_best,w_judge_threshold = fisher(train_data1,train_data2)
    correct_num = 0
    correct_rate = 0
    for i in range(len(test_data1)):
        if np.dot(w_best,test_data1[i])>w_judge_threshold:
            correct_num+=1
    for i in range(len(test_data2)):
        if np.dot(w_best,test_data2[i])<w_judge_threshold:
            correct_num+=1
    correct_rate = correct_num/(len(test_data1)+len(test_data2))
    print("correct_num",correct_num)
    print("correct_rate",correct_rate)
    visualize_data_and_fisher_line(train_data1,train_data2,w_best,w_judge_threshold)