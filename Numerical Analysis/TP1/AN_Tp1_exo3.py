# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:37:17 2022

@author: cyril
"""

#*********************************1 pivot partiel***************************************

import scipy.linalg.lapack as lap
import numpy as np
import scipy.linalg


def init_matrix(n):
    dl = np.random.rand(n-1)
    d = np.random.rand(n)
    du = np.random.rand(n-1)
    return dl,d,du

def init_rhs(n,nrhs):
    b = np.random.rand(n,nrhs)
    return b



# xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx xxxxxxx
#  d[0]    du[0]     0       0       0       0       0       0       0
#  dl[0]   d[1]    du[1]     0       0       0       0       0       0
#    0     dl[1]   d[2]    du[2]     0       0       0       0       0
#
#   0        0       0       0       0    dl[n-4] d[n-3]  du[n-3]    0
#   0        0       0       0       0       0    dl[n-3] d[n-2]  du[n-2]
#   0        0       0       0       0       0       0    dl[n-2]  d[n-1]

def my_dgtsv(dl,d,du,b):
    n=len(d)
    du2=np.zeros(n-2)
    for i in range(n-2):
        if(np.abs(dl[i])>np.abs(d[i])):
            dl[i],d[i] = d[i],dl[i]
            d[i+1],du[i] =du[i],d[i+1]
            du[i+1],du2[i] = du2[i],du[i+1]
            tmp = np.copy(b[i,:])
            b[i,:] = np.copy(b[i+1,:])
            b[i+1,:] = np.copy(tmp)
        piv = dl[i]/d[i]
        d[i+1] -= piv*du[i]    
        du[i+1] -= piv*du2[i]
        b[i+1,:] -= piv*b[i,:]     
    
    i=n-2
    if(np.abs(dl[i])>np.abs(d[i])):
        dl[i],d[i] = d[i],dl[i]
        d[i+1],du[i] =du[i],d[i+1]
        tmp = np.copy(b[i,:])
        b[i,:] = np.copy(b[i+1,:])
        b[i+1,:] = np.copy(tmp)
    piv = dl[i]/d[i]
    d[i+1] -= piv*du[i]
    b[i+1,:] -= piv*b[i,:]     
    b = my_remontee(d,du,du2,b)    
    #b = my_remontee2(d,du,du2,b)    
    return du2,d,du,b

# xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
#   d[0]     du[0]   du2[0]     0        0        0        0        0        0
#    0        d[1]   du[1]    du2[1]     0        0        0        0        0
#    0        0       d[2]     du[2]     0        0        0        0        0
#
#    0        0        0        0        0        0      d[n-3]   du[n-3] du2[n-3]
#    0        0        0        0        0        0        0       d[n-2] du[n-2]
#    0        0        0        0        0        0        0         0     d[n-1]


def my_remontee(d,du,du2,b):
    n=len(d)
    b[n-1,:] /= d[n-1]
    i=n-2
    b[i,:] = (b[i,:]-du[i]*b[i+1,:])/d[i]  
    for i in range(n-3,-1,-1):
        b[i,:] = (b[i,:]-du[i]*b[i+1,:]-du2[i]*b[i+2,:])/d[i]
    return b

def my_remontee2(d,du,du2,b):
    A = np.diag(d,0) + np.diag(du,1) + np.diag(du2,2)
    xx = np.linalg.solve(A,b)
    return xx



n=4   #2000
nrhs = 2 # 200
verbose = 0

print(' ')
dl,d,du = init_matrix(n)
print('dl= ', dl)
print('d= ', d)
print('du= ', du)
print(' ')

b = init_rhs(n,nrhs)
print('b=',b)
print(' ')

A = np.diag(d,0) + np.diag(du,1) + np.diag(dl,-1)
print('A=', A)

dl_1 = np.copy(dl)
d_1 = np.copy(d)
du_1 = np.copy(du)
b_1 = np.copy(b)
du2_1,d_1,du_1,x_1,info = lap.dgtsv(dl_1,d_1,du_1,b_1)
if(verbose>0):
    print("du2,d,du")
    print(du2_1)
    print(d_1)
    print(du_1)
    
dl_2 = np.copy(dl)
d_2 = np.copy(d)
du_2 = np.copy(du)
b_2 = np.copy(b)
du2_2,d_2,du_2,x_2 = my_dgtsv(dl_2,d_2,du_2,b_2)
if(verbose>0):
    print("du2,d,du")
    print(du2_2)
    print(d_2)
    print(du_2)
print(' ')

P,L,U=scipy.linalg.lu(A)
if(verbose>0):
    print(P)
    print(L)
    print(U)
print(' ')    

xx = np.linalg.solve(A,b)
if(verbose>0):
    print('xx = ', xx)
    print('x_1 = ', x_1)
    print('x_2 = ', x_2)

print(' ')
print(np.linalg.norm(np.dot(A,xx)-b))
print(np.linalg.norm(np.dot(A,x_1)-b))
print(np.linalg.norm(np.dot(A,x_2)-b))
print(np.linalg.norm(x_1-x_2))




#************************** 2 pivot partiel *******************************
def recherche_pivot(A, b, j):
    p = j
    for i in range(j+1, A.shape[0]):
        if abs(A[i, j]) > abs(A[p, j]):
           p = i
        if p != j:
           b[p, j] = b[j, p]
           A[p, j] = A[j, p]
 
def triangular_sup(A, b, j):
    for i in range(j+1, A.shape[0]):
        b[[i]] = b[[i]] - (A[i, j] / A[j, j]) * b[[j]]
        A[[i]] = A[[i]] - (A[i, j] / A[j, j]) * A[[j]] #[[i]]: i-ème ligne
       
def descente(A, b):
    for j in range(A.shape[1] - 1):
        recherche_pivot(A, b, j)
        triangular_sup(A, b, j)

def remontee(A, b):
    n = A.shape[0]
    for i in range(n-2, -1,-1):
        for j in range(i+1, n):
            b[i] = b[i] - np.sum( (A[i,j]/A[j,j])*b[j])
       
def resol_systeme(A, b):
    for k in range(b.shape[0]):
        b[k] = b[k] / A[k, k]
    return b

def gauss_partiel(A, b):
    U = A.copy()
    y = b.copy()
    descente(U, y)
    remontee(U, y)
    return  resol_systeme(U, y)


#____________________________EXO_3____________________________________________#

du2,d,du,x,info = scipy.linalg.lapack.dgtsv(dl,d,du,b)
xx = np.linalg.solve(A,b)

