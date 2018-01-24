import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def J(theta, X, y, learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = learningRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    
    return np.sum(first - second) / len(X) + reg
    
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    error = sigmoid(X * theta.T) - y
    
    grad = (error.T * X) / len(y) + learningRate / len(y) * theta
    grad[0, 0] = np.sum(error.T * X[:,0]) / len(y)
    return np.array(grad).ravel()
    
def onevsall(X, y, label_num, learningRate):
    '''
    计算 opt_theta 
    '''
    opt_theta = np.zeros((label_num, X.shape[1])) # 10 * 401
    
    for i in range(1, label_num+1):
        print(i)
        theta = np.zeros(X.shape[1]).reshape((1,401)) # 1 * 401
        y_i = np.array([1 if i==j else 0 for j in y]).reshape((X.shape[0], 1))

        fmin = minimize(fun=J, x0=theta, args=(X, y_i, learningRate), method='TNC', jac=gradient)

        opt_theta[i-1,:] = fmin.x.reshape((1, 401))
    
    return opt_theta
    
def predict(x, opt_theta):
    '''
    x          -->  1 x 401
    opt_theta  -->  10 x 401
    '''
    opt_theta = np.matrix(opt_theta).reshape((10, 401))
    x = np.matrix(x).reshape((1, 401))
    
    result = sigmoid(opt_theta * x.T)
    result = list(result)
    max_v = max(result)
    index = result.index(max_v) + 1
    
    return index

def accuracy(opt_theta, X, y):
    correct = 0
    for i in range(len(X)): 
        if predict(X[i,:], opt_theta) == y[i,0]:
            correct += 1
    print('accuracy:{}'.format(correct / len(X)))

data = loadmat('ex3data1.mat')
X = data['X']
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1) # 5000 * 401
y = data['y']

opt_theta = onevsall(X, y, 10, 1)
accuracy(opt_theta, X, y)  # --> 训练集正确率94.48%



