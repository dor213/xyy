import numpy as np

w1= np.array([[5,37],[7,30],[10,35],[11.5,40],[14,38],[12,31]])
w2= np.array([[35,21.5],[39,21.7],[34,16],[37,17]])
w1_label = np.ones(len(w1))
w2_label = np.ones(len(w2))*-1

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
if __name__ == '__main__':
    w1_covariance = calculate_covariance(w1)
    w2_covariance = calculate_covariance(w2)
    print("w1_covariance",w1_covariance)
    print("w2_covariance",w2_covariance)
    w1_mean = calculate_average(w1)
    w2_mean = calculate_average(w2)
    print("w1_mean",w1_mean)
    print("w2_mean",w2_mean)
    sw= w1_covariance + w2_covariance
    print("sw",sw)
    sw_inverse = np.linalg.inv(sw)
    w_best = np.dot(sw_inverse,w1_mean-w2_mean)
    print("w_best",w_best)
    w_best_T = w_best.T
    w_judge_threshold = np.dot(w_best_T,w1_mean+w2_mean)/2
    print("w_judge_threshold",w_judge_threshold)
    