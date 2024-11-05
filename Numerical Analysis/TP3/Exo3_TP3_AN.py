import math
import numpy as np

A=[[10,7,8,7]
   ,[7,5,6,5]
   ,[8,6,10,9]
   ,[7,5,9,10]]

def qrA(B,e):
    Ak=B
    i=0
    while np.max(np.abs(np.tril(Ak)-np.diag(np.diag(Ak,0))))>e and i < 1000:
        Q,R=np.linalg.qr(Ak)
        Ak=np.dot(R,Q)
        i+=1
    return Ak

#print(qrA(A,10**(-10)))

M=np.random.randint(10,size=(4,4))


M_t=np.transpose(M)

P=np.dot(M_t,M)


D1=np.diag([6,5,2,1])
D2=np.diag([5,1,2,6])
D3=np.diag([5,3,2,3])
A1=np.dot(np.dot(P,D1),np.linalg.inv(P))
A2=np.dot(np.dot(P,D2),np.linalg.inv(P))
A3=np.dot(np.dot(P,D3),np.linalg.inv(P))

# print(qrA(A1,10**(-10)))
# print(qrA(A2,10**(-10)))
# print(qrA(A3,10**(-10)))

D4=[[5,0,0,0],
    [0,2,0,0],
    [0,0,0,1],
    [0,0,1,0]]

A4=np.dot(np.dot(P,D4),np.linalg.inv(P))
print(qrA(A4,10**(-10)))


print(np.linalg.eig(A4)[0])
print(np.linalg.eig(qrA(A4,10**(-10)))[0])


