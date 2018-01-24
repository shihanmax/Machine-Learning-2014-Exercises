import os
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
  
def sigmoid_gradient(z):
    return np.multiply(1 - sigmoid(z), sigmoid(z))
  
def forward_propagate(X, theta1, theta2):
    '''
    X: 5000 x 400
    theta1: 25 x 401
    theta2: 10 x 26
    '''
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1) # 5000 x 401，为input增加一个值为1的feature
    z2 = a1 * theta1.T # 5000 x 25
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) # 5000 x 26
    z3 = a2 * theta2.T # 5000 x 10
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h

def back_propagate(params, input_size, hidden_size, num_labels, X, y, learningRate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size+1)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, hidden_size+1)))
    
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    cost = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    
    for i in range(m):
        first = np.multiply(-y[i,:], np.log(h[i,:]))
        second = np.multiply(1 - y[i,:], np.log(1 - h[i,:]))
        cost += np.sum(first - second)
    cost /= m
    #为代价函数添加正则化项
    cost += (float(learningRate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    
    # back propagation
    for t in range(m):
        a1t = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        ht = h[t,:]
        yt = y[t,:]
        
        d3t = ht - yt
        
        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))
        
        delta1 += d2t[:,1:].T * a1t
        delta2 += d3t.T * a2t
    
    delta1 /= m
    delta2 /= m
    
    #增加正则化项
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learningRate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learningRate) / m
    
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return cost, grad
    
# 数据准备
data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']

# 对 label 进行 one-hot 编码
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y) # 5000 x 10

input_size = 400
hidden_size = 25
num_labels = 10
learningRate = 1
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25 # 随机初始化参数
m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)
# 结束 数据准备

if not os.path.exists('fminx.npy'):
    fmin = minimize(fun=back_propagate, 
                    x0=params, 
                    args=(input_size, hidden_size, num_labels, X, y_onehot, learningRate), 
                    method='TNC', 
                    jac=True, 
                    options={'maxiter':250, 'disp':True})
    
    np.save('fminx', fmin.x)
    opt_theta = fmin.x
else:
    opt_theta = np.load('fminx.npy')
    
theta1 = np.matrix(np.reshape(opt_theta[:hidden_size * (input_size + 1)], (hidden_size, input_size+1)))
theta2 = np.matrix(np.reshape(opt_theta[hidden_size * (input_size + 1):], (num_labels, hidden_size+1)))
    
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_prediction = np.array(np.argmax(h, axis=1) + 1)
correct = [1 if i==j else 0 for i,j in zip(y, y_prediction)]
accuracy = sum(correct) / X.shape[0] * 100

print('accuracy: {}%'.format(round(accuracy, 3)))






