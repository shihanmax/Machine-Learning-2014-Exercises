import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

def PCA(X):
    X = (X - X.mean()) / X.std()
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    
    U, S, V = np.linalg.svd(cov)
    
    return U, S, V

def compress_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)
    
def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)
    
mat1 = loadmat('data/ex7data1.mat')
data1 = pd.DataFrame(mat1.get('X'), columns=['X1', 'X2'])
data1 = np.array(data1)
U, S, V = PCA(data1)
Z = compress_data(data1, U, 1)

X_c = recover_data(Z, U, 1)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(list(data1[:, 0]), list(data1[:,1]))
ax[1].scatter(list(X_c[:,0]), list(X_c[:, 1]))
plt.show()