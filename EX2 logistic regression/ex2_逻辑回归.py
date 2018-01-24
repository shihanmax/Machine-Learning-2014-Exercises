import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def J(theta, X, y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / len(y)

# def gradient(theta, X, y):
    # grid = sum(np.multiply(sigmoid(X * theta.T) - y, X)) / len(y)
    # return grid.ravel()
    
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    return grad
    
# 不懂为什么上面两个函数，使用第一个执行 fmin_tnc 线性搜索失败

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
    
data = pd.read_csv('ex2data1.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])

'''
# 数据分布绘图
pos = data[data['Admitted'].isin([1])]
neg = data[data['Admitted'].isin([0])]

plt.scatter(pos['Exam1'], pos['Exam2'], s=5, c='g', label='Admitted')
plt.scatter(neg['Exam1'], neg['Exam2'], s=5, c='r', label='Not admitted')
plt.legend()
plt.xlabel('Score of Exam 1')
plt.ylabel('Score of Exam 2')
plt.show()
'''

data.insert(0, 'Ones', 1)
cols = data.shape[1]

X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

#分割训练集和验证集
X_train = X[:70]
X_valid = X[70:]
y_train = y[:70]
y_valid = y[70:]

result = opt.fmin_tnc(func=J, x0=theta, fprime=gradient, args=(X_train, y_train)) # 用最优化函数寻找最优theta
# gradient(theta, X, y)
opt_theta = result[0] # 最优theta
accuracy(opt_theta, X_valid, y_valid) # 86.7% 正确率（在训练集上验证模型正确率并不合适）







