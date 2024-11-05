import numpy as np
import pivot as pv


#______________________EXO_1_____________________________#

A=np.array([[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]], dtype=float)
print('A :',A)
print('')
t=np.linalg.eigvals(A)
print('Valeurs propres de A :',t)

b=np.array([[32],[23],[33],[31]], dtype=float)

def sdp(A):
    print('')
    t=np.linalg.eigvals(A)
    for i in t:
        if i <= 0:
            return(print('La matrice nest pas symétrique définie positive'))
    return(print('La matrice est symetrique défnie positive'))

sdp(A)

def sol(A,b):
    x=np.linalg.solve(A,b)
    return x


""""
#################TEST######################
"""
print('')
x=sol(A,b)
print('La solution x du système Ax=b est x= : \n',x,)
print('')


X1=pv.gauss_partiel(A, b)
print('X1 :', X1)

def dx(x,y):
    t=[]
    for i in range(len(x)):
        t.append(x[i] - y[i])
    return t




"""
###################Perturbation###################
"""

print('')
print('PERTURBATION')
print('')

db=[0.1,-0.1,0.1,-0.1]

def vect_perturb(y,y1):
    x=[]
    for i in range(len(y)):
        x.append(y[i]+y1[i])
    return x

b1=vect_perturb(b, db)

print('b perturbé :', b1)

x1=sol(A,b1)
print('x perturbé :',x)

print('')
dx=dx(x, x1)
print('dx :', dx)

#Verif Conditionnement#
def cond(A):
    x=np.linalg.norm(A)
    y=np.linalg.norm(np.linalg.inv(A))
    return x*y

print('')
cond=cond(A)
print("cond(A) =",cond)

def norm(x):
    return(np.linalg.norm(x))

DX=norm(dx) / norm(x)
print('DX :', DX)

DB=norm(db)/norm(b)
print('DB :', DB)


print('')

def verif_cond(cond,DX,DB):
    print('Inegalité ',DX <= cond*DB )
    

    
verif_cond(cond, DX, DB)



