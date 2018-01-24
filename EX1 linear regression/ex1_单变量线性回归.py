import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# start function definations
def J(X, y, theta):
    '''
    计算代价函数
    '''
    inner = np.power((X*theta.T - y), 2)
    return np.sum(inner)/(2*len(y))

def computeLoss(X, y, theta):
    '''
    计算模型值和真实值的差
    '''
    return X*theta.T - y
    
def gridientDescent(X, y, theta, alpha, iters):
    '''
    向量化批量梯度下降
    '''
    nx = len(y)
    cost = []
    for i in range(iters):
        print('iteration {}'.format(i))
        dj_theta = X.T * computeLoss(X, y, theta)# 2by1 matrix
        dj_theta = dj_theta.T / nx
        
        theta -= alpha * dj_theta
        cost.append(J(X, y, theta))
    
    plt.plot(range(iters), cost)
    plt.show()
    
    return theta, cost[-1]
# end function definations
    
 
# start data preparation

# 用pandas库读取csv
data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])

# 绘图查看数据分布
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()

# 插入一列x 
data.insert(0, 'Ones', 1)

#初始化变量
cols = data.shape[1] # 列数目
X = data.iloc[:, 0:cols-1] # 两个参数分别为行索引，列索引。所有行，列减掉最后一列
y = data.iloc[:, cols-1:cols] # 所有行，最后一列

#将变量转化为 matrix，并初始化theta
X = np.matrix(X)
y = np.matrix(y)
theta = np.matrix(np.array([0, 0]), dtype='float64')

# 查看一下它们的维度
print(X.shape, y.shape, theta.shape)
# end data preparation


# start training
alpha = 0.01
iters = 1000
theta, cost = gridientDescent(X, y, theta, alpha, iters)
print(theta, cost)
# end training


# start plot
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0, 0] + theta[0, 1] * x
plt.scatter(data.Population, data.Profit, s=5)
plt.plot(x, f, color='red')
plt.show()
# end plot
