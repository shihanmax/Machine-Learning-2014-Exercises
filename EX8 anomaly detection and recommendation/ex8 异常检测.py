import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
import time

def gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma

def choose_threshold(pval, yval):
    '''
    验证集用来选择 threshold，此过程只用验证集，不用训练集
    '''
    best_eps = 0
    F1_max = 0
    for eps in np.arange(pval.min(), pval.max(), 0.001):
        predictions = pval < eps
        tp = np.sum(np.logical_and(predictions == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(predictions == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(predictions == 0, yval == 1)).astype(float)
        try:
            Precision = tp / (tp + fp)
            Recall = tp / (tp + fn)
            F1 = 2*Precision*Recall / (Precision+Recall)
        except:
            F1 = 0
            
        if F1 > F1_max:
            best_eps = eps
            F1_max = F1
            
    return best_eps, F1_max

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
p = np.array([i*j for i,j in zip(p[:,0], p[:,1])]).reshape(-1,1)

pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])
pval = np.array([i*j for i,j in zip(pval[:,0], pval[:,1])]).reshape(-1,1)

epsilon, _ = choose_threshold(pval, yval)
anonis = np.where(p < epsilon)
plt.scatter(X[:,0], X[:,1], s=10, c='b')
plt.scatter(X[anonis[0], 0], X[anonis[0], 1], s=10, c='r')
plt.show()