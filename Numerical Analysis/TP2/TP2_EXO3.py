import numpy as np
import pivot as pv
import matplotlib.pyplot as plt

def norm(x):
    return(np.linalg.norm(x))

def dx(x,y):
    t=[]
    for i in range(len(x)):
        t.append(x[i] - y[i])
    return t

N=3
h=1/(N+1)
def Mat(N):
    h=1/(N+1)
    d=(2/(h*h))*np.ones(N)
    d1= (-1/(h*h))*np.ones(N-1)
    A=np.diag(d,0)+np.diag(d1,1)+np.diag(d1,-1)
    return A

def f(x):
    y=(np.pi*np.pi)*np.sin(np.pi*x)
    return (y)

A=Mat(N)

x=np.linspace(0,1,N+2)

B=f(x)

X=np.linalg.solve(A,B[1:-1])
print('X :', X, '\n')

def cond(A):
    x=np.linalg.norm(A)
    y=np.linalg.norm(np.linalg.inv(A))
    return x*y

cond_n=[]
y=[]
X=[]
cond_F=[]

for n in range(100,1001):
    A=Mat(n)
    x=np.linspace(0,1,n+2)
    B=f(x)
    X=np.linalg.solve(A,B[1:-1])
    cond_A=cond(A)
    #cA.append(cond_A)
    dB=np.random.rand(n+2)*0.1
    
    B1=B+dB
    
    X1=np.linalg.solve(A,B1[1:-1])
    
    dX=dx(X,X1)
    cond_f= (norm(B)/norm(dB))*(norm(dX)/norm(X))
    cond_F.append(cond_f)

    cond_n.append(cond(A))
    y.append(n)

plt.plot(y,cond_n, label="Cond(A_n)")
plt.plot(y,cond_F, label="Cond_f(A_n)")
plt.legend()
plt.show




    


    
