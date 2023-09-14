import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return x*np.cos(0.25*np.pi*x)

def gradient(x):
    return np.cos(0.25*np.pi*x) - 0.25*np.pi*x*np.sin(0.25*np.pi*x)

def gradient_descent(x, learnning_rate, iterations):
    result = [] 
    for i in range(iterations):
        x = x - learnning_rate*gradient(x)
        result.append(x)
    return result

def gradient_descen_with_RMSProp(x, learnning_rate, iterations, gamma):
    result = []
    G = 0
    epsilon = 1e-8
    for i in range(iterations):
        G = gamma*G + (1-gamma)*(gradient(x)**2)
        x = x - learnning_rate*gradient(x)/(np.sqrt(G) + epsilon)
        result.append(x)
    return result

def gradient_descent_with_momentum(x, learnning_rate, iterations, gamma):
    result = []
    v = 0
    for i in range(iterations):
        v = gamma*v + learnning_rate*gradient(x)
        x = x - v
        result.append(x)
    return result

def gradient_descent_with_adagrad(x, learnning_rate, iterations):
    result = []
    G = 0
    epsilon = 1e-8
    for i in range(iterations):
        G += gradient(x)**2
        x = x - learnning_rate*gradient(x)/(np.sqrt(G) + epsilon)
        result.append(x)
    return result

def gradient_descent_with_adam(x, learnning_rate, iterations, gamma1, gamma2):
    result = []
    m = 0
    G = 0
    epsilon = 1e-6
    for i in range(iterations):
        m = gamma1*m + (1-gamma1)*gradient(x)
        G = gamma2*G + (1-gamma2)*(gradient(x)**2)
        alpha = learnning_rate*np.sqrt(1-gamma2**(i+1))/(1-gamma1**(i+1))
        x = x - alpha*m/(np.sqrt(G) + epsilon)
        result.append(x)
    return result

def visualize_result_and_function(result, iterations):
    x = np.linspace(-4, 4, 100)
    y = f(x)
    plt.plot(x, y, label='f(x)')
    plt.plot(result, f(np.array(result)), 'bo', label='result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x = -4
    iterations = 100
    learnning_rate = 0.1
    gamma = 0.9
    gamma1 = 0.9
    gamma2 = 0.999
    result = gradient_descent(x, learnning_rate, iterations)
    visualize_result_and_function(result, iterations)
    result = gradient_descen_with_RMSProp(x, learnning_rate, iterations, gamma)
    visualize_result_and_function(result, iterations)
    result = gradient_descent_with_momentum(x, learnning_rate, iterations, gamma)
    visualize_result_and_function(result, iterations)
    result = gradient_descent_with_adagrad(x, learnning_rate, iterations)
    visualize_result_and_function(result, iterations)
    result = gradient_descent_with_adam(x, learnning_rate, iterations, gamma1, gamma2)
    visualize_result_and_function(result, iterations)