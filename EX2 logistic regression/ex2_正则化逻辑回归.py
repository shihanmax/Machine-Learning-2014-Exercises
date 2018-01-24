import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model
import time

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def J_with_reg(theta, X, y, learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = learningRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    
    return np.sum(first - second) / len(X) + reg

# def gradient_with_reg1(theta, X, y, learningRate):
    
    # grad = sum(np.multiply(sigmoid(X * theta.T) - y, X)) / len(y)
    # return grad
    
def gradient_with_reg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + (learningRate / len(X)) * theta[:,i]

    return grad
    
# ！！！不懂为什么上面两个函数，使用第一个执行 fmin_tnc 会报错

def predict(theta, X):
    theta = np.matrix(theta)
    x = np.matrix(X)
    p = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in p]

def accuracy(theta, X, y):
    predictions = predict(theta, X)
    correct = [1 if p==t else 0 for p,t in zip(predictions, y)]
    print(correct)
    print('accuracy: {}'.format(sum(correct) / len(correct)))
    
data = pd.read_csv('ex2data2.txt', header=None, names=['Test1', 'Test2', 'Accepted'])

'''
# 数据分布绘图
pos = data[data['Accepted'].isin([1])]
neg = data[data['Accepted'].isin([0])]

plt.scatter(pos['Test1'], pos['Test2'], s=10, c='g', marker='o', label='Accepted')
plt.scatter(neg['Test1'], neg['Test2'], s=10, c='r', marker='x', label='Not accepted')
plt.legend()
plt.xlabel('test1 score')
plt.ylabel('test2 score')
plt.show()
# 结束 数据分布绘图
'''

# 特征映射 feature mapping
degree = 5
x1 = data['Test1']
x2 = data['Test2']
data.insert(3, 'Ones', 1)

for i in range(1, degree):
    for j in range(i):
        data['F'+str(i)+str(j)] = np.power(x1, i-j) * np.power(x2, j)
        
data.drop('Test1', axis=1, inplace=True)
data.drop('Test2', axis=1, inplace=True)
# 结束 特征映射

# 数据和参数的调整
cols = data.shape[1]

X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(11)

learningRate = 1
# 结束 数据和参数的调整

print(J_with_reg(theta, X, y, learningRate)) # 初始theta的损失函数
print(gradient_with_reg(theta, X, y, learningRate)) # 初始theta的带正则化项的梯度

result = opt.fmin_tnc(func=J_with_reg, x0=theta, fprime=gradient_with_reg, args=(X, y, learningRate)) # 训练过程；用最优化函数寻找最优theta
opt_theta = result[0] # 最优theta
print('opt_theta:{}'.format(opt_theta))
accuracy(opt_theta, X, y) # 最优theta的正确率


# 调用 sklearn 线性回归包
model = linear_model.LogisticRegression()
model.fit(X, y.ravel())
print('accuracy of sklearn：{}'.format(model.score(X, y)))
# 结束 调用 sklearn 线性回归包

