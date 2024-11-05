import numpy as np
import matplotlib.pyplot as plt

N=10
h=1/(N+1)

def construct_A_b():
    A = np.array([[10, 7, 8, 7],
                  [7, 5, 6, 5],
                  [8, 6, 10, 9],
                  [7, 5, 9, 10]], dtype=float)
   
    b = np.array([[32], [23], [33], [31]], dtype=float)  
    return A,b


def construct_An ():
    d0= 2/(h**2)*np.ones(N-2)
    d1=(-1/h**2)*(np.ones(N-3))
    d2=(-1/h**2)*(np.ones(N-3))
    A = np.diag(d0,0) + np.diag(d1,-1) + np.diag(d2,+1)
    return A

def f(x):
    return (np.pi)*(np.pi)*(np.sin(2*(np.pi)*x))

def bn():
    bn=[]
    for i in range(1,N-1):
        bn.append(f(i*h))
    return bn

def resoud_Xn(A,b):
    Xn=np.linalg.solve(A,b)
    return Xn

'''
def jacob(A, b, eps, x0, itermax):
    D = np.diag(np.diag(A))
    E = -np.triu(A) + D
    F = -np.tril(A) + D
    #print('A = ', D - E - F)
    D = la.inv(D)
    B = np.dot(D, E + F)
    bb = np.dot(D, b)
   
    x = la.solve(A,b)
    #print('exact sol: ', x)
   
    k = 0
    xk = x0
    for k in range(itermax):
        #k +=1
        xk = np.dot(B, xk) + bb
        #print(xk)
        r = max(abs(b - np.dot(A, xk)))
        e = max(abs(x - xk))
        if r <= eps:
           #print(k, '  ', x, '  ', xk)
           break      
        xk = xk
   
    print('xJ=', xk)
    BB = la.matrix_power(B, itermax)
    rspec =  la.norm(BB, 'fro')
    rspec = rspec**(1./itermax)
    #rspec = max(abs(la.eigvals(B)))    
    return r, e, rspec
'''

def Jacobi(A,b,eps,x0, itermax):
    D = np.diag(np.diag(A))
    E = - np.tril(A) + D
    F = - np.triu(A) + D
    b1 = np.dot(np.linalg.inv(D), b)
    B = np.dot(np.linalg.inv(D), E + F)
   
    x = np.linalg.solve(A,b1)
   
    k=0
    x_k = x0
    for k in range(itermax):
        x_k= np.dot(B, x_k) + np.dot(np.linalg.inv(D),b)
       
        r = max(abs(b - np.dot(A, x_k)))
        e = max(abs(x - x_k))
       
        if r <= eps:
            break
       
    rspec = np.linalg.norm(np.linalg.matrix_power(B, itermax))**(1/itermax)
       
    return r, e, rspec


#############___________MAIN________________###########################
'''
An=np.array(construct_An(), dtype=float)
print(An)

print("")

bn=np.array(bn(), dtype=float)
print(bn)

print("")

Xn=resoud_Xn(An, bn)
print(Xn)
'''




A = np.array([[10, 7, 8, 7],
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]], dtype=float)

b = np.array([[32], [23], [33], [31]], dtype=float)


x0  = [[1.], [1.], [1.], [1.]]

eps = 1.e-6

for k in range(1, 100):
    r_j, e_j, rspec_j = Jacobi(A, b, eps, x0, k)

   
    plt.semilogy(k, e_j, 'r*')

#plt.semilogy(k, e_j, 'r*', label="erreur Jacobi")
plt.axhline(y=rspec_j, color='r', label='rayon spectral Jacobi')
plt.xlabel("k")
plt.legend()
plt.grid()
plt.show()





def Gauss(A,b,eps,x0, itermax):
    D = np.diag(np.diag(A))
    E = - np.tril(A) + D
    F = - np.triu(A) + D
    b2 = np.dot(np.linalg.inv(D-E),b)
    B = np.dot(np.linalg.inv(D-E),F)
   
    x = np.linalg.solve(A,b2)
   
    k=0
    x_k = x0
    for k in range(itermax):
        x_k= np.dot(B, x_k) + np.dot(np.linalg.inv(D-E),b)
       
        r = max(abs(b - np.dot(A, x_k)))
        e = max(abs(x - x_k))
       
        if r <= eps:
            break
       
    rspec = np.linalg.norm(np.linalg.matrix_power(B, itermax))**(1/itermax)
       
    return r, e, rspec


A = np.array([[10, 7, 8, 7],
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]], dtype=float)

b = np.array([[32], [23], [33], [31]], dtype=float)


x0  = [[1.], [1.], [1.], [1.]]

eps = 1.e-6

for k in range(1, 100):
    r_j, e_j, rspec_j = Gauss(A, b, eps, x0, k)

   
    plt.semilogy(k, e_j, 'b*')

#plt.semilogy(k, e_j, 'r*', label="erreur Jacobi")
plt.axhline(y=rspec_j, color='b', label='rayon spectral Gauss')
plt.xlabel("k")
plt.legend()
plt.grid()
plt.show()



def SOR(A,b,eps,x0, itermax,w):
    D = np.diag(np.diag(A))
    E = - np.tril(A) + D
    F = - np.triu(A) + D
    b1 = np.dot(np.linalg.inv(D-np.dot(w,E)),np.dot(b,w))
    B = np.dot(
        np.linalg.inv(
            np.dot(D,1/w)-E)
        ,(F+np.dot(
            D,(1-w)/w)))
   
    x = np.linalg.solve(A,b1)
   
    k=0
    x_k = x0
    for k in range(itermax):
        x_k= np.dot(B, x_k) + b1
       
        r = max(abs(b - np.dot(A, x_k)))
        e = max(abs(x - x_k))
       
        if r <= eps:
            break
       
    rspec = np.linalg.norm(np.linalg.matrix_power(B, itermax))**(1/itermax)
       
    return r, e, rspec

w=0.001
for k in range(1, 100):
    r_j, e_j, rspec_j = SOR(A, b, eps, x0, k,w)

   
    plt.semilogy(k, e_j, 'g*')

#plt.semilogy(k, e_j, 'r*', label="erreur Jacobi")
plt.axhline(y=rspec_j, color='g', label='rayon spectral SOR')
plt.xlabel("k")
plt.legend()
plt.grid()
plt.show()
