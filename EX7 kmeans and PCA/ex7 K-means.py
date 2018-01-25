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
    return np.random.rand(cent_num, X.shape[1])
    
def find_next_centroids(X, centroids):
    m = X.shape[0]
    cent_num = centroids.shape[0]
    groups = []
    for i in range(cent_num):
        groups.append([])
    for i in range(m):
        closet_dis = 100000
        for j in range(cent_num):
            distance_point_i_to_centroid_j = calc_EucDistance(X[i,:], centroids[j, :])
            if distance_point_i_to_centroid_j < closet_dis:
                closet_dis = distance_point_i_to_centroid_j
                closet_cen = j

        groups[closet_cen].append(X[i, :])
        
    new_centroids = []

    for group in groups:
        group = np.array(group)
        new_centroids.append((sum(group)/len(group)).reshape(1, X.shape[1]) if not len(group)==0 else np.random.rand(X.shape[1]).reshape(1, X.shape[1]))
    new_centroids = np.array(new_centroids)
    return new_centroids.reshape(cent_num, X.shape[1]), groups
    
def find_closet_centroids(X, initial_centroids):
    while True:
        print('epoch with \n{}\n'.format(initial_centroids))
        next_centroids, groups = find_next_centroids(X, initial_centroids)
        if calc_EucDistance(initial_centroids, next_centroids) == 0:
            return next_centroids, groups
        initial_centroids = next_centroids

def total_distance(centroids, groups):
    loss = 0
    for i in range(len(groups)):
        for point in groups[i]:
            loss += calc_EucDistance(centroids[i,:], point)
    return loss

def choose_best_group_number(X):
    '''
    K-means 聚类时，不确定类的个数时可以计算各个K值下总距离，
    本例给出的数据中，由肘部法则得到最佳类个数为3
    '''
    loss = []
    for i in range(2, 10):
        initial_centroids = initialize_centroids(X, i)
        centroids, groups = find_closet_centroids(X, initial_centroids)
        loss.append(total_distance(centroids, groups))
        
    plt.plot(range(2, 10), loss)
    plt.show()
    
mat1 = loadmat('data/ex7data1.mat')
mat2 = loadmat('data/ex7data2.mat')

data1 = pd.DataFrame(mat1.get('X'), columns=['X1', 'X2'])
data2 = np.array(pd.DataFrame(mat2.get('X'), columns=['X1', 'X2']))
choose_best_group_number(data2)


