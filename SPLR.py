# Unsupervised feature selection via self-paced learning and low-redundant regularization
# input:
# X:data matrix with n rows and d columns
# alpha,lambda1,lambda2,lambda3:balance parameters
# beta,k,mu:parameters related to self-paced learning
# maxIter:maximum iteration number
# K:dimension of the subspace
# output:
# W:projection matrix with d rows and K columns
# index:sorted index of the l2-norm of rows of matrix W
# obj:values of the objective function during the iteration
import math
import numpy.matlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

def initializek(X,W,H):
    [n, d] = X.shape
    L=0
    for i in range(n):
        temp=X[i,...]-X[i,...]*W*H
        L = L + np.linalg.norm(temp,ord=2) ** 2

    return 1/math.sqrt(L/n)

def updateV(X,W,H,k,beta):
    [n, d] = X.shape
    for i in range(n):
        temp=X[i,...]-X[i,...]*W*H #混合正则化
        L=np.linalg.norm(temp,ord=2) ** 2
        v=np.empty((1,n))
        if (L>=1/(K**2)):
            v[0,i]=0
        elif (L<=1/(k+1/beta)**2):
            v[0,i]=1
        else :v[0,i]=beta*(1/math.sqrt(L)-K)
    return v

def updateU(W):
    [d,k]=W.shape
    U=np.matlib.zeros((d,d))
    for i in range(d):
        temp0=W[i,...].getA1()*W[i,...].getA1()
        temp=sum(temp0)**(3/4)
        U[i,i]=1/max(temp,np.spacing(1))
    return U

def SPLR(X,alpha,lambda1,lambda2,lambda3,beta,mu,maxIter,K):
    [n,d]=X.shape #原始矩阵
   #Initialize W and H
    W=np.matlib.ones((d,K))
    H=np.matlib.rand((K,d))
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    X = min_max_scaler.fit_transform(X.T)
    X=X.T
    #calculate the similarity between features

    featuresX = normalize(X)
    S=np.matmul(featuresX.T,featuresX)
    S=S-np.diag(np.diag(S)) #S（特征空间的相似矩阵）
    #print(S.shape)

    #calculate the similarity between samples

    samplesX = normalize(X.T)
    samplesX = samplesX.T
    Z =np.matmul(samplesX , samplesX.T)  #Z(数据空间的相似矩阵)

    #calculate the Laplacian matrix L
    #L=D-Z D是一个对角矩阵，Z为图G的邻接矩阵

    L = np.diag(sum(Z, 2)) - Z #L(数据空间的拉普拉斯矩阵)
    iter = 0

    #Initialize k

    k = initializek(X, W, H);
    #print(k)
    obj = np.zeros((maxIter,n))
    while(iter<=maxIter):
        #update v
        v = updateV(X, W, H, k, beta)
        #update G

        v=np.sqrt(v)
        G=np.diag(v)*X
        #update H
        H=H*(np.dot(W.T,(np.dot(G.T,G)))/(np.dot(W.T,(np.dot(G.T,G)))*W*H+np.spacing(1)))
        #update V
        V=updateU(W)
        #update W

        up=G.T.dot(G).dot(H.T)+lambda2*X.T.dot(Z).dot(X).dot(W)+lambda3*W
        down=G.T.dot(G).dot(W)*H.dot(H.T)+alpha*V.dot(W)+lambda1*S.dot(W)*np.ones((K,K))+lambda2*X.T.dot(L+Z).dot(X).dot(W)+lambda3*W.dot(W.T).dot(W)+np.spacing(1)
        W=W*up/down

        #calculate the value of the objective function
        temp1=G-G*W*H
        temp2=W.T*W-np.eye(K,K)
        temp3=np.trace(temp1*temp1.T)+alpha*np.trace(W.T*updateU(W)*W)+beta**2/sum(v+beta*k)+lambda1*np.trace(S.T*W*np.ones((K,K))*W.T)+lambda2*np.trace(W.T*X.T*L*X*W)+lambda3/2*np.trace(temp2*temp2.T)
        obj[iter]=temp3

        #update k//k:1/eta
        k=k/mu
        #the stop criteria
        if iter>1:
            if (abs(obj[iter-1]-obj[iter])/abs(obj[iter-1]).all()<10**-3).all():
                break

        iter=iter+1
    #calculate the score


if __name__ == '__main__':
    #读取数据集
    X=pd.read_excel(r'D:\python1\SPLR\COIL20.xlsx',sheet_name=0)
    #print(X)
    K=X.columns.__len__() #K>=N
    alpha=1
    lambda1=1
    lambda2=1
    lambda3=1
    beta = 1
    mu=1
    maxIter = 10
    SPLR(X,alpha,lambda1,lambda2,lambda3,beta,mu,maxIter,K)


