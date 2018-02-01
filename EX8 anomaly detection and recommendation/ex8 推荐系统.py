from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os

def J(params, Y, R, num_of_features, learningRate):
    '''
    在协同过滤算法中，需要同时优化 theta 和 X，
    在使用 optimize() 函数时，需要传入平坦化的参数，
    这里将 X 和 theta 合并为 params，在 J 函数内进行计算时，
    需要将二者从 params 中提取并 reshape。
    '''
    Y = np.matrix(Y) # num_of_movies * num_of_usrs
    R = np.matrix(R)
    num_of_movies = Y.shape[0]
    num_of_usrs = Y.shape[1]
    
    # 分割 params
    X = np.matrix(params[:num_of_movies*num_of_features].reshape(num_of_movies, num_of_features))
    theta = np.matrix(params[num_of_movies*num_of_features:].reshape(num_of_usrs, num_of_features))
    
    # 计算代价函数
    err = np.multiply(X * theta.T - Y, R) # shape == R.shape
    sqr_err = np.power(err, 2)
    J = 0.5 * np.sum(sqr_err)
    
    # 加上正则化项
    J += 0.5 * learningRate * np.sum(np.power(X, 2))
    J += 0.5 * learningRate * np.sum(np.power(theta, 2))
    
    # X 和 theta 的梯度
    X_grad = err * theta
    theta_grad = err.T * X
    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))
    
    return J, grad

def load_movie_idxs():
    '''
    读取ex8 movie_ids.txt
    返回 dic : {idx:moive}
    '''
    movies = {}
    with open('data/movie_ids.txt') as f:
        for line in f.readlines():
            idx_movie = line.split(' ')
            
            idx = int(idx_movie[0]) - 1
            movie = ' '.join(idx_movie[1:]).rstrip()
            movies[idx] = movie
    return movies

# 原始数据，包括：部分用户对部分电影的评分情况    
data = loadmat('data/ex8_movies.mat')
Y = data['Y'] #1682(nm) X 943(nu) 评分数据
R = data['R'] #1682 X 943 是否评分

# 可视化评分数据
# plt.imshow(Y)
# plt.show()

# 读取参数（实际训练过程并未用到，用于对比训练结果？？）
params_data = loadmat('data/ex8_movieParams.mat')
X = params_data['X'] # 1682 *10
Theta = params_data['Theta'] # 943 * 10
movies_dic = load_movie_idxs()

# 添加自己的电影评分数据
my_rate = np.zeros((X.shape[0], 1))
my_rate[0] = 4
my_rate[6] = 3
my_rate[15] = 1
my_rate[22] = 3
my_rate[25] = 5
my_rate[38] = 5
my_rate[41] = 2
my_rate[42] = 1
my_rate[230] = 5
my_rate[570] = 5
my_rate[595] = 4
my_rate[1013] = 5
my_rate[1275] = 2
my_rate[1386] = 2
my_rate[1555] = 1
my_rate[1681] = 5
Y = np.append(Y, my_rate, axis=1)
R = np.append(R, my_rate !=0, axis=1)

# 参数预设
movies = Y.shape[0] # 电影数量
usrs = Y.shape[1] # 用户数量
features = 10 # 电影特征数量
learningRate = 10

# 参数随机生成
X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(usrs, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

# 对评分数据均值归一化
Y_mean = np.zeros((movies, 1))
Y_norm = np.zeros((movies, usrs))

for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Y_mean[i] = Y[i, idx].mean()
    Y_norm[i, idx] = Y[i, idx] - Y_mean[i]

# 避免反复训练
if not os.path.exists('data/fminx.npy'):
    fmin = minimize(fun=J, x0=params, args=(Y_norm, R, features, learningRate), method='CG', jac=True, options={'maxiter':100, 'disp':True})
    np.save('data/fminx', fmin.x)
    fit_params = fmin.x
else:
    fit_params = np.load('data/fminx.npy')

# 此处的 X 和 theta 都是训练好的数据，可用来预测用户对未评分电影的评分
X = np.matrix(fit_params[:movies*features].reshape(movies, features)) # 1682 * 10
theta = np.matrix(fit_params[movies*features:].reshape(usrs, features)) # 944 * 10
my_theta = theta[943, :] # #943 用户的特征向量（用于预测对各种电影的喜爱程度（每个电影有 10 个特征））
my_prediction = X * my_theta.T + Y_mean # 1682 * 1 即是用户# 943对 1682 部电影的预测评分





