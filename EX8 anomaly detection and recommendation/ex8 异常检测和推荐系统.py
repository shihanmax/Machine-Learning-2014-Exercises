import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

def gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma

def choose_threshold(pval):
    '''
    验证集用来选择 threshold，此过程只用验证集，不用训练集
    '''
    
data = loadmat('data/ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

# plt.figure(1)

# plt.subplot(211) # 两行（2）一列（1）第一个（1）
# plt.scatter(X[:,0], X[:,1], s=6, c='r')
# plt.title('X')

# plt.subplot(212)
# plt.scatter(Xval[:,0], Xval[:,1], s=6, c='g')
# plt.title('Xval')
# plt.show()

mu, sigma = gaussian(X)

p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

pval = np.zeros((Xval.shape[0], Xval.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])

