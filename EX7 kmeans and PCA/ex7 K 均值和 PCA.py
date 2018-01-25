import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

def calc_EucDistance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
    
def plot_data(data, fit_reg=False):
    plt.scatter(data[:, 0], data[:, 1], s=10)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def initialize_centroids(X, cent_num):
    # cent_num -> 聚类中心点的个数
    max = X.max().max()
    min = X.min().min()
    return np.random.randint(min, max, size=(cent_num, X.shape[1]))
    
def find_next_centroids(X, centroids):
    m = X.shape[0]
    cent_num = centroids.shape[0]
    new_centroids = np.array([[0,0], [0,0], [0,0]])
    groups = [0, 0, 0]
    for i in range(m):
        closet_dis = 100000
        closet_cen = 0
        for j in range(cent_num):
            distance_point_i_to_centroid_j = calc_EucDistance(X[i,:], centroids[j, :])
            if distance_point_i_to_centroid_j < closet_dis:
                closet_dis = distance_point_i_to_centroid_j
                closet_cen = j
                
        if isinstance(groups[closet_cen], int):
            groups[closet_cen] = X[i, :]
        else:
            groups[closet_cen] = np.vstack((groups[closet_cen], X[i, :]))
    
    new_centroids = []
    for group in groups:
        new_centroids.append(sum(group)/len(group))

    new_centroids = np.array(new_centroids)
        
    return new_centroids
    
def find_closet_centroids(X, initial_centroids):
    old, new =  
            
            
mat1 = loadmat('data/ex7data1.mat')
mat2 = loadmat('data/ex7data2.mat')

data1 = pd.DataFrame(mat1.get('X'), columns=['X1', 'X2'])
data2 = np.array(pd.DataFrame(mat2.get('X'), columns=['X1', 'X2']))
initial_centroids = initialize_centroids(data2, 3)
new_cen = find_colset_centroids(data2, initial_centroids)
new_cen2 = find_colset_centroids(data2, new_cen)