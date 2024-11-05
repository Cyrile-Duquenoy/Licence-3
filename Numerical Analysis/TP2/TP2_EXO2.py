import numpy as np

N=10

#Construction de D# Question 1
def Diag(n):
    d=np.random.rand(n)*10
    d=np.sort(d)
    A=np.diag(d)
    return A

D=Diag(N)
print('D :', D)
print("")

#################################################

#Construction matrice carée aléatoire#
def Mat(n):
    A=np.random.rand(n,n)*10
    if(np.linalg.det(A)==0):
        print('Matrice non inversible')
        return Mat(n)
    return A


#Retourne Q de la décomposition QR#
def QR(X):
    Q, R = np.linalg.qr(X)
    return Q

# Y Matrice aléatoire
Y=Mat(N)
print('Y :', Y)
print('')

# On sort Q de la décompo. QR de Y
Q=QR(Y)
print('Q :', Q)

#####################################################

#Fait le produit Q D Q^t#
def Mat_3(Q,D):
    return np.dot( Q , np.dot(D, np.transpose(Q)))

print('')

A2=Mat_3(Q,D)
print('A=QD :', A2)

print('')

######################################################

def col_n(Q):
    b=[]
    for i in range(N):
        b.append(Q[i][N-1])
    return b

b=col_n(Q)

print('b', b)
print('')

x=np.linalg.solve(np.linalg.inv(A2),b)
#x=np.dot(A2,b)
print('x :', x)
print("")

def col_1(Q):
    db=[]
    for i in range(N):
        db.append(Q[i][0])
    return db

db=col_1(Q)
print('db :', db)
print('')

b1=[]
for i in range(N):
    b1.append(b[i] + db[i])
print('b + db :', b1)

print('')

x2=np.dot(A2,b1)
print('x+dx', x2)
print('')

dx=[]
for i in range(N):
    dx.append(x2[i] - x[i])
    
print('dx',dx)
print('')

##########################################################################

def cond(A):
    x=np.linalg.norm(A)
    y=np.linalg.norm(np.linalg.inv(A))
    return x*y

def norm(x):
    return(np.linalg.norm(x))

#################Probleme : Egalité FAUSSE !!!!!!!!!!!!!!! ###############


cond_A=cond(A2)

print('cond(A)',cond_A)
print('')

a=norm(dx)/norm(x)
c=norm(db)/norm(b)

c1=norm(dx)/norm(x) * norm(b)/norm(db)

print('a',a)
print('')

print('c*cond(A)',c*cond_A)
print('')

print('c1',c1)



    