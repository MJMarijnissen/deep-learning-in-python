# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:16:39 2019

@author: Kubus
"""
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt


Nclass = 500
D=2 #dimensions of input
M=3 #hiddden layer size
K=3 #number of classes
X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])
   
Y= np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N= len(Y)

T=np.zeros((N,K))
for i in range(N):
    T[i, Y[i]] = 1

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

X, T = shuffle(X, T)
Ntrain = int(0.7*len(X))
Xtrain, Ttrain = X[:Ntrain], T[:Ntrain]
Xtest, Ttest = X[Ntrain:], T[Ntrain:]

model = MLPClassifier(hidden_layer_sizes=(2,8), max_iter=2000) #hidden layer + number of neutrons

model.fit(Xtrain, Ttrain)

train_accuracy = model.score(Xtrain, Ttrain)
test_accuracy = model.score(Xtest, Ttest)
print("train accuracy:", train_accuracy, "test accuracy", test_accuracy)

Xpredict1 = np.array([0.2,-1.8])
Xpredict2 = np.array([2.2,2.2])
Xpredict3 = np.array([4,-4])
Xpredict = np.vstack([Xpredict1, Xpredict2, Xpredict3])

predict = model.predict(Xpredict)

print(predict)