import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def load_and_show(file, show=True):
    '''
    载入数据并查看数据分布
    '''
    data_mat = loadmat(file)
    data = pd.DataFrame(data_mat['X'], columns=['X1', 'X2'])
    data['y'] = data_mat['y']
    
    if show:
        pos = data[data['y'].isin([1])]
        neg = data[data['y'].isin([0])]
        plt.scatter(pos['X1'], pos['X2'], s=20, marker='x', label='Positive')
        plt.scatter(neg['X1'], neg['X2'], s=20, marker='o', label='Negative')
        plt.legend()
        plt.show()
        
    return data
 
def choose_best_C_and_gamma():
    '''
    C和gamma对SVM模型的影响
    '''
    data_mat3 = loadmat('data/ex6data3.mat')
    X = data_mat3['X']
    Xval = data_mat3['Xval']
    y = data_mat3['y']
    yval = data_mat3['yval']

    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

    best_score = 0
    best_params = {'C':None, 'gamma':None}
    for C in C_values:
        for gamma in gamma_values:
            svc = svm.SVC(C=C, gamma=gamma)
            svc.fit(X, y)
            score = svc.score(Xval, yval)
            if score > best_score:
                best_score = score
                best_params['C'] = C
                best_params['gamma'] = gamma
    print(best_score, best_params)
    
data1 = load_and_show('data/ex6data1.mat', False)
data2 = load_and_show('data/ex6data2.mat', False)    

# 在数据集1上线性SVM模型的正确率变化曲线与C值的关系
score = []
for c in np.arange(1, 101, 5):
    svc = svm.LinearSVC(C=c, loss='hinge', max_iter=1000)
    svc.fit(data1[['X1', 'X2']], data1['y'])
    score.append(svc.score(data1[['X1', 'X2']], data1['y']))
    
plt.plot(np.arange(1, 101, 5), score)
plt.show()

# 在数据集2上使用非线性SVM的表现
svc = svm.SVC(C=100, gamma=10, probability=True)
svc.fit(data2[['X1', 'X2']], data2['y'])
print(svc.score(data2[['X1', 'X2']], data2['y']))

# SVM在垃圾邮件识别上的应用
spam_train = loadmat('data/spamTrain.mat')
spam_test = loadmat('data/spamTest.mat')
X = spam_train['X']
y = spam_train['y']
Xtest = spam_test['Xtest']
ytest = spam_test['ytest']
# shape of above: (4000, 1899), (4000, 1), (1000, 1899), (1000, 1)

svc = svm.SVC()
svc.fit(X, y)
accuracy = svc.score(Xtest, ytest)
print('accuracy:{}%'.format(accuracy*100))