import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def J(theta, X, y):
    '''
    代价函数
    '''
    theta = np.matrix(theta).reshape(X.shape[1], 1)
    inner = np.power(X*theta - y, 2)
    return np.sum(inner) / (X.shape[0] * 2)

def regularized_J(theta, X, y, l=1):
    '''
    带正则化项的代价函数 
    '''
    m = X.shape[0]
    reg_term = l / (2*m) * np.sum(np.power(theta, 2))
    return J(theta, X, y) + reg_term
    
def gradient(theta, X, y):
    '''
    代价函数的梯度 
    '''
    theta = np.matrix(theta).reshape(X.shape[1], 1)
    grad = np.sum(np.multiply((X*theta - y), X), axis=0) / len(y)
    grad = np.array(grad).ravel()

    return grad

def regularized_gradient(theta, X, y, l=1):
    '''
    带正则化项的代价函数的梯度
    '''
    m = X.shape[0]
    reg_term = theta.copy()
    reg_term[0] = 0
    reg_term = reg_term * l / m

    grad = gradient(theta, X, y) + reg_term.ravel()
    grad = np.array(grad).ravel()
    # print('type and size')
    # print(type(grad))
    # print(grad.shape)
    # time.sleep(100)
    return grad
    
def linear_regression(theta, X, y, fun, jac, l=1):
    '''
    返回最优 theta
    '''
    return opt.minimize(fun=fun, x0=theta, args=(X, y, l), method='TNC', jac=jac, options={'disp':True}).get('x')

def poly_data(*args, power):
    '''
    args 可以是多个 array
    为各个 array 添加最高为 power 次的高阶项并将它们返回 
    '''
    def poly_it(arg):
        for i in range(2, power+1):
            arg = np.insert(arg, i, np.power(arg[:,1], i).ravel(), axis=1)
        return arg
    return [poly_it(arg) for arg in args]
    
def normalize_data(arr):
    '''
    返回 (arr - arr的平均值) / arr的标准差
    '''
    nor_arr = arr.copy()
    for i in range(1, arr.shape[1]):
        mean = np.mean(arr[:,i])
        standard_diff = np.std(arr[:,i])
        nor_arr[:, i] = arr[:, i] - mean
        nor_arr[:, i] = arr[:, i] / standard_diff
    return nor_arr

def learning_curve(theta, X, y, Xval, yval, l=1):
    '''
    绘制学习曲线
    '''
    train_cost = []
    cv_cost = []
    for i in range(1, X.shape[0]+1):
        theta = linear_regression(theta, X[:i, :], y[:i], regularized_J, regularized_gradient, l)
        tc = regularized_J(theta, X[:i, :], y[:i])
        cv = regularized_J(theta, Xval, yval)
        train_cost.append(tc)
        cv_cost.append(cv)
        
    plt.plot(np.arange(X.shape[0]), train_cost, label='train_cost', c='g')
    plt.plot(np.arange(X.shape[0]), cv_cost, label='cv_cost', c='b')
    plt.plot(np.arange(X.shape[0]), [0]*X.shape[0], label='zero', c='r')
    plt.legend(loc=1)
    plt.show()
    
# 读取数据
data = sio.loadmat('ex5data1.mat')
(X, y, Xtest, ytest, Xval, yval) = map(np.ravel, [data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data['yval']])

# 查看数据分布
df = pd.DataFrame({'water_level':X, 'flow':y})
sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
plt.show()

# 为所有的 X 添加一列
X, Xtest, Xval = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in [X, Xtest, Xval]]

# 将数据矩阵化
X = np.matrix(X).reshape(-1, 2)
y = np.matrix(y).reshape(-1, 1)
Xtest = np.matrix(Xtest).reshape(-1, 2)
ytest = np.matrix(ytest).reshape(-1, 1)
Xval = np.matrix(Xval).reshape(-1, 2)
yval = np.matrix(yval).reshape(-1, 1)

# 多项式扩展
X, Xtest, Xval = poly_data(X, Xtest, Xval, power=8)

# 均一化数据
X, Xtest, Xval = map(normalize_data, [X, Xtest, Xval])

# 定义 theta
theta = np.matrix(np.ones(X.shape[1])).reshape(X.shape[1], 1)

# 绘制学习曲线,可尝试不同的 l (lambda)
learning_curve(theta, X, y, Xval, yval, l=1)





