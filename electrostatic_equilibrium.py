import numpy as np
import matplotlib.pyplot as plt
from newton_raphson import *
from random import uniform
import numpy.polynomial.legendre as L


###question 0

def E(X):
    ##Return the value of E(x1,...,xn)
    E=0
    N=len(X)
    for i in range(N):
        E+=np.log(abs(X[i]+1))+np.log(abs(X[i]-1))
        for j in range(N):
            if j!=i:
                if X[i]!=X[j]:
                    E+=np.log(abs(X[i]-X[j]))
                else:
                    return -7 ## To avoid a divsion by 0
    return E


def grad_E(X):
    ## Return the vector grad E as it is described in the subject
    n, m = np.shape(X)
    assert (m == 1)
    res = np.zeros([n,1])
    for i in range(n):
        s = 0
        for j in range(n):
            if (i != j):
                s += 1. / (X[i, 0] - X[j, 0])
        res[i, 0] = 1. / (1 + X[i]) + 1. / (X[i] - 1) + s
    return res

def F(X):
    return grad_E(X)

###quest1


def Jacobian_E(Y):
    ## Return the Jacobian matrix of delta_E
    N = np.shape(Y)[0]
    J=np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            if i==j:
                J[j][i]=-1/(Y[i]+1)**2 - 1/(Y[i]-1)**2
                for k in range(N):
                    if k!=j:
                        J[j][i]+=-1/(Y[i]-Y[k])**2
            else:
                J[i][j]=1/(Y[i]-Y[j])**2
    return J

def J(X):
    return Jacobian_E(X)

## Two global lists to the display of the curve
norme_F=[]
iterations=[]


def Newton_Raphson_curve(f, J, U0, N, eps):
    ## The Newton-Raphson algorithm modified to allow the display of the curve ||F(X)||
    global norme_F
    global iterations
    norme_F=[]
    iterations=[]
    """
    Solve nonlinear system F=0 by Newton-Raphson's method.
    J is the Jacobian of F. At input, U0 is the starting 
    position of the algorithm. The iteration continues
    until ||F|| < eps or until N iterations are reached.
    """
    F_value = f(U0)
    U = U0
    F_norm = np.linalg.norm(F_value, ord=2)
    iteration_counter = 0
    norme_F.append(F_norm)
    iterations.append(iteration_counter)
    while abs(F_norm) > eps and iteration_counter < N:
        V = np.linalg.solve(J(U), -F_value)
        U = U + V
        F_value = f(U)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

        norme_F.append(F_norm)
        iterations.append(iteration_counter)

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
        
    return U, iteration_counter


def Newton_Raphson_backtracking_curve(f, J, U0, N, eps, alpha):
    ## The Newton-Raphson algorithm with backtracking modified to allow the display of the curve ||F(X)||
    global norme_F
    global iterations
    norme_F=[]
    iterations=[]
    """
    Solve nonlinear system F=0 by Newton-Raphson's method.
    J is the Jacobian of F. At input, U0 is the starting 
    position of the algorithm. The iteration continues
    until ||F|| < eps or until N iterations are reached.
    There is a backtracking to reach the solution faster.
    """
    F_value, U = f(U0), U0
    F_norm = np.linalg.norm(F_value, ord=2)
    iteration_counter = 0

    while abs(F_norm) > eps and iteration_counter < N:
        V = np.linalg.lstsq(J(U), -F_value, rcond=None)[0]
        nxt_F_norm = np.linalg.norm(f(U + V), ord=2)
        i = 0    
        while nxt_F_norm >= F_norm :
            print('BACKTRACKING')
            i += 1
            nxt_F_norm = np.linalg.norm(f(U + alpha ** i * V), ord=2)
            
        U = U + alpha ** i * V
        F_value = f(U)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

        norme_F.append(F_norm)
        iterations.append(iteration_counter)
        
    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
    return U, iteration_counter



## Initialisation of  Ten chosen charges
U0=np.zeros((10,1))
U0[0]=-0.86684278
U0[1]=0.86088026
U0[2]=0.80889216
U0[3]=-0.98098176
U0[4]=0.68707341
U0[5]=0.27329905
U0[6]=-0.07208807
U0[7]=0.6864963
U0[8]=-0.11970087
U0[9]=-0.1899953


#we could have imported the function from  newton_raphson.py but this version allows to draw the curve 

def Newton_Raphson_with_backtracking(f, J, U0, N, epsilon):
    global norme_F
    global iterations
    norme_F=[]
    iterations=[]
    F_value, U = f(U0), U0
    F_norm = np.linalg.norm(F_value, ord=2)
    iteration_counter = 0
    for i in range(N):
        fu = f(U)
        na = np.linalg.norm(fu)
        if (na < epsilon):
            return U
        ju = J(U)
        V = np.linalg.lstsq(ju,-fu)[0]
        if (np.linalg.norm(f(U+V)) - na >= 0):
            V = (1.0/3.0)*V
        U = U + V
        F_value = f(U)
        F_norm = np.linalg.norm(F_value, ord=2)
        iteration_counter += 1

        norme_F.append(F_norm)
        iterations.append(iteration_counter)
    return U, iteration_counter


U,iteration_counter=Newton_Raphson_curve(F, J, U0, N=100, eps=1e-8)

def test_equilibrium():
    #a function that shows the electrostatic equilibrium with 10 charges with and without backtracking
    U,iteration_counter=Newton_Raphson_curve(F, J, U0, N=100, eps=1e-8)
    print("Test with 10 charges, initialisation of U0 : ",U0.transpose())
    print("Final positions of the charges : ",U.transpose())
    plt.plot(iterations,norme_F,label="without backtracking")
    plt.title("Electrostatic equilibrium with 10 charges")
    Newton_Raphson_with_backtracking(F, J, U0, N=100, epsilon=1e-8)
    #plt.plot(iterations,norme_F,label="using backtracking")
    plt.xlabel("Number of iterations")
    plt.ylabel("||F(X)||")
    plt.title("Electrostatic equilibrium with 10 charges with and without backtracking")
    plt.semilogy()
    plt.legend()
    plt.show()

def position_real_axis():
    #a function that shows the final position ofthe charges on the real axis
    plt.figure(num=2,figsize=(7,1.5))
    print("Final positions on the real axis")
    plt.title("Position of the charges")
    plt.xlabel("x axis")
    plt.plot([min(U),max(U)],[0,0],color="red",label="Real axis")
    plt.plot(U,[0]*len(U),'o',color="yellow",label="Charge")
    plt.legend()
    plt.show()

def energy_one_charge_position():
    ## The plot of the curve for one charge
    size=50

    O=np.linspace(-0.99,0.99,size)
    V=np.zeros((size,1))
    for k in range(size):
       V[k]=E([O[k]])

    print("Graph describing the evolution the electrostatic energy of one charge")
    plt.title("Electrostatic energy of one charge")
    plt.ylabel("Energy")
    plt.xlabel("Position of the charge")
    plt.plot(O,V)
    plt.show()


def mirror(A):
    n = A.size
    for i in range(n//2): 
        tmp = A[i]
        A[i] = A[n-i-1]
        A[n-i-1] = tmp
    return A

#Plot Legendre polynomials and equilibrium positions
def add_plot(X, lbl, clr, type='o'):
    Peq = Newton_Raphson(grad_E, Jacobian_E, X, 100, 1e-8)
    n= Peq.size
    for i in range(n):
        plt.plot(Peq[i,0],0,type, color=clr)
    
    c = [0]*(n+2)
    c[n+1] = 1
    
    d = L.legder(c)
    P = L.leg2poly(d)

    P = mirror(P)
    Poly = np.poly1d(P)
    x = np.linspace(-1,1,100)
    y = Poly(x)
    plt.plot(x, y, label=lbl, color=clr)


#Electrostatic Equilibrium Test

def elec_equ_test():

    A = np.matrix([[0.2]])

    B = np.matrix([[0.5],
                   [0.6]])

    C = np.matrix([[0.4],
                   [-0.5],
                   [0.7]])

    D = np.matrix([[0.4],
                   [-0.4],
                   [0.5],
                   [0.6]])
    add_plot(A, 'n=1', 'r')
    add_plot(B, 'n=2', 'y')
    add_plot(C, 'n=3', 'b')
    add_plot(D, 'n=4', 'g')
    plt.plot([-1,1], [0,0], 'k')
    plt.axis([-1,1,-4,4])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Legendre polynomials and equilibrium positions")
    plt.show()

if __name__ == '__main__':
    test_equilibrium()
    position_real_axis()
    energy_one_charge_position()
    elec_equ_test()


