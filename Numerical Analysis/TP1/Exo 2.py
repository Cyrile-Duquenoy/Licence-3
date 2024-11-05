# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:21:47 2022

@author: cyril
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from math import log
from math import exp
from math import sqrt

N=3

a=np.random.rand(N)
b=np.random.rand(N-1)
c=np.random.rand(N-1)

def LU(a,b,c):  #Complexité en O(n)
    N=np.size(a)
    U,L=[]*2,[]*2
    f=[]
    g=[]
    h=[]
    #Construction du U
    for k in range(len(b)):
        h.append(b[k])  
    for k in range(len(a)):
        if k==0:
            g.append(a[k])
        else :
            g.append(a[k] - ( b[k-1] * (c[k-1] / g[k-1])))
    U=np.diag(g)+np.diag(h,1)
    #Construction de L
    x=[1]*N
    for k in range(len(c)):
        f.append(  (c[k]) / (g[k])   )
    L=np.diag(x)+np.diag(f,-1)
    return L,U

L,U=LU(a,b,c)
A=np.diag(a)+np.diag(b,1)+np.diag(c,-1)

#print(A, np.dot(L,U))

print('A-LU : ', A-np.dot(L,U))
print()

y=np.random.rand(N)


def remonte (U, y ) :
    g=[]
    h=[]
    for k in range(0,N-1):
        h.append(U[k][k+1])
    #del h[0]
    print('h :', h)
    print('')
    for k in range(len(y)):
        g.append(U[k][k])
        
    n = np.size(g)
    x = [1] * n
    x[n-1] = y[n-1] / g[n-1]
    for i in range  (n-2 , -1 , -1) :
        x[i] = ( y[i] - x[i+1] * h[i] ) / g[i]
    return x


print('U', U)
x=remonte(U,y)
print('x :',x)
print('')


#__________________________________Application________________________________#
print('APPLICATION')
print('')


h=1/(N-1)

def Laplacien(a,b,c,y):
    print('y :', y)
    L,u=LU(a,b,c)
    Y=np.dot(np.linalg.inv(L),y)
    n=np.size(a)
    g=[]
    for i in range(0,n):
        g.append(U[i][i])
    h=[]
    for k in range(0,n-1):
        h.append(U[k][k+1])
    X=remonte(U,Y)
    X1=np.dot(L,U)
    print('Véif Calcul')
    print('On tombe bien sur le y choisi')
    print('Vérif LUX=Y :', np.dot(X1,X))
    return X

#-------------------------------------TEST------------------------------------#

y=[1,2,3]
d=(2/(h*h))*np.ones(N)
d1= (-1/(h*h))*np.ones(N-1)
d2= (-1/(h*h))*np.ones(N-1)

print(Laplacien(d,d1,d2,y))
print('')

#___________________________Inverse_Matrice _Sans _Numpy______________________#
def inverse_L(L):
    n=np.size(L[0])
    f=[]
    for i in range (0,n-1):
        f.append(L[i+1][i])
    I=np.zeros((n,n))+np.diag(np.ones(n))
    for j in range(1,n):
        I[j]=I[j]-[i*f[j-1] for i in I[j-1]]
    print("vérification de l'inverse")
    print(np.dot(L,I))
    return I

def inverse_U(U):
    n=np.size(U[0])
    g=[]
    for i in range (0,n):
        g.append(U[i][i])
    h=[]
    for k in range (0,n-1):
        h.append(U[k][k+1])
    I=np.zeros((n,n))+np.diag(np.ones(n))
    I[n-1]=[r/g[n-1] for r in I[n-1]]
    for j in range(n-2,-1,-1):
        I[j]=np.array([p/g[j] for p in I[j]])-[i*(h[j]/g[j]) for i in I[j+1]]
    print("vérification de l'inverse")
    print(np.dot(U,I))
    return I

def inverse_A(a,b,c):
    n=np.size(a)
    A=np.zeros((n,n))+np.diag(a)+np.diag(b,1)+np.diag(c,-1)
    L,U = LU(a,b,c)
    inv_u=inverse_U(U)
    inv_l=inverse_L(L)
    inv_a=np.dot(inv_u,inv_l)
    print('A')
    print(A)
    print("vérification de l'inverse")
    print(np.dot(A,inv_a))
    print()
    print("l'inverse de A est")
    return inv_a

print(inverse_A(d,d1,d2))


