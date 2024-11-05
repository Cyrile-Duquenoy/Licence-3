#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:56:41 2022

@author: d15018891
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
            g.append(a[k] - (b[k-1])*( c[k-1] / g[k-1] ))
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

def remonte(U,y):
    temp=[]
    g=[]
    h=[]
    for k in range(len(y)):
        g.append(U[k][k]) 
    for k in range(len(y-1)):
        h.append(U[k-1][k])
        
    #temp = (x_N, x_{N-1}, ..., x_2, x_1)
    for k in range(len(y)):
        if k==0:
            temp.append(y[k] / g[k])
        else :
            temp.append((y[k] - (h[k] * temp[k-1]))  / (g[k]))
            
    #x=(x_1,x_2, ..., x_N)
    x=[]
    for i in range(len(y)):
        x.append(temp[N-i-1]) 
        
    return x
    
print(remonte(U,y))
print()


#_____________________________Application_____________________________________#

h=1/N
y=np.random.rand(N)
def Mat(N):
    #h=1/N
    d=(2/(h*h))*np.ones(N)
    d1= (-1/(h*h))*np.ones(N-1)
    A=np.diag(d,0)+np.diag(d1,1)+np.diag(d1,-1)
    return A

d=(2/(h*h))*np.ones(N)
d1= (-1/(h*h))*np.ones(N-1)
d2= (-1/(h*h))*np.ones(N-1)

A=Mat(N)
print('y :', y)
print('')

print('A :', A)
print()
L,U=LU(d,d1,d2)
print('L :',L)
print()


print('U :', U)
print()
X=remonte(U, y)
print('X remonte :', X)
print()

print('Vérif',np.dot(L,U),'', A)
print()

print('X verif',np.dot(np.linalg.inv(A),y))
print()

#--------------INVERSE----------------------#

def inverse(M,I):
    for fd in range(len(M)): #Pour chaque colonne de M
        fdScaler = 1.0/M[fd][fd] #Scalaire qui va diviser notre ligne
        for j in range(len(M)): #pour chaque colonne de M
            M[fd][j] *= fdScaler #On multiplie les coeff de la ligne par notre scalaire
            I[fd][j] *= fdScaler #De même pour l'identité
        for i in list(range(len(M)))[0:fd] + list(range(len(M)))[fd+1:]: #On parcours [fd+1,fd+2,...,2*fd]
            crScaler = M[i][fd] 
            for j in range(len(M)):
                M[i][j]=M[i][j] - crScaler * M[fd][j] 
                I[i][j]=I[i][j] - crScaler * I[fd][j]
    return I

print(inverse(np.dot(L,U) , np.dot(L,U) ))



