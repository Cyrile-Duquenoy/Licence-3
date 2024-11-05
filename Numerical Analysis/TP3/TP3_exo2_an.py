import math
import numpy as np

def methode_puissance(A,k):
    n=np.size(A[0])
    x=[1]
    l=[0]*(n-1)
    x = x+l
    X=[x/np.linalg.norm(x)]
    for i in range (0,k):
        X.append((np.dot(A,X[i]))/(np.linalg.norm(np.dot(A,X[i]))))
    return np.dot(np.dot(A,X[k-1]),X[k-1])

A1=[[10,7,8,7]
   ,[7,5,6,5]
   ,[8,6,10,9]
   ,[7,5,9,10]]

print(methode_puissance(A1, 2000))


n=10

h=1/(1+n)

An=np.diag([2/h**2]*n,0)+np.diag([-1/h**2]*(n-1),1)+np.diag([-1/h**2]*(n-1),-1)

print(methode_puissance(An, 2000))