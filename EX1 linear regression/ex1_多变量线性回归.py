import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# start 函数定义
def computeCost(X, y, theta):
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
    
def gridientDescent(X, y, theta, alpha, iters, showCost=False):
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
        cost.append(computeCost(X, y, theta))
    if showCost:
        plt.plot(range(iters), cost)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Error vs. Training Epoch')
        plt.show()
    return theta, cost[-1]
    
def normalEquation(X, y):
    '''
    正规方程与梯度下降法相比，不需要选择学习率alpha，不需要迭代，但一般仅适用于
    特征数量 n 小于 10000 的训练过程（因矩阵求逆的时间复杂度为O(n^3)），而且它只
    适用于线性模型，不适合逻辑回归等模型。
    '''
    theta = np.linalg.inv(X.T * X) * X.T * y
    return theta
# end 函数定义
 
# start 数据准备
# 用pandas库读取csv
data = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])

# 标准化数据
data = (data - data.mean()) / data.std()
# 绘图查看数据分布
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()
print(data.describe())
# 插入一列x
data.insert(0, 'Ones', 1)

#初始化变量
cols = data.shape[1] # 列数目
X = data.iloc[:, 0:cols-1] # 所有行，列减掉最后一列，两个参数分别为行索引和列索引。
y = data.iloc[:, cols-1:cols] # 所有行，最后一列

#将变量转化为 matrix，并初始化theta
X = np.matrix(X)
y = np.matrix(y)

theta = np.matrix(np.zeros(len(X.T)), dtype='float64')

# 查看一下它们的维度
print(X.shape, y.shape, theta.shape)
# end 数据准备

# start 训练
alpha = 0.01
iters = 1000
theta, cost = gridientDescent(X, y, theta, alpha, iters, showCost=True)
print('梯度下降：\ntheta  {}\ncost  {}'.format(theta, cost))
# end 训练

# start 正规方程
print('正规方程：\ntheta  {}'.format(normalEquation(X, y).T))
# end 正规方程