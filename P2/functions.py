import numpy as np
from scipy import exp, sparse,sin,cos ,pi
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
import matplotlib.pyplot as plt


def twopBVP(fvec, alpha, beta, L, N):
    # Two point boundry value problem solver.
    # fvec is a vector of the function values f from y´´ = f(x)
    # in the N points from 0 to L.

    dx = L/(N+1)

    subp = np.ones(N-1)
    mid = np.ones(N) * (-2)

    diagonals = [subp, mid, subp]

    T = sparse.diags(diagonals, [-1, 0, 1])

    # Create array for rhs of equation
    b = np.zeros(N)
    b[0] = -alpha/(dx*dx)
    b[-1] = -beta/(dx*dx)

    b = dx * dx * np.add(b, fvec)

    y = spsolve(T.tocsc(), b)

    # Append boundary values
    y = np.insert(y, 0, alpha)
    y = np.append(y, beta)

    return y


def test_f(x):
    # apply the function f to the array x
    # x assumed to be column np array [[a], [b]...]

    return exp(-x)

def test_fsin(x):
    # apply function fsin to array x
    # x assumed to be column np array
    return sin(x)


def exact_sin(x, alpha ,beta ,L):
    return alpha -sin(x) + x*(beta + sin(L)-alpha)/L


def exact_f(x, alpha, beta, L):
    return exp(-x) + alpha - 1 + (x/L) * (beta - alpha + 1 - exp(-L))


def vshBvp(alpha, beta, length, N):
    errors = np.array([])
    step_sizes = np.array([])

    for k in N:
        step_size = length / (k + 1)
        x = np.linspace(step_size, length - step_size, k)
        fvec = test_f(x)

        y = twopBVP(fvec, alpha, beta, length, k)

        x = np.insert(x, 0, 0)
        x = np.append(x, length)

        y_exact = exact_f(x, alpha, beta, length)

        err = norm(y[-2]-y_exact[-2])

        errors = np.append(errors, err)
        step_sizes = np.append(step_sizes, step_size)

    plt.loglog(step_sizes, errors)
    plt.grid()
    plt.show()
    
    
def func_q(x):
    #returns 1d array with -50kN
    return -5*10**4 * np.ones(len(x))


def func_I(x,L):
     return (3-2*np.power(cos(x*pi/L),12))* 10**-3
    

def beamSolve(fvec, L, N):
    # Boundary values alpha and beta are assumed to be the same for the beam problem
    m0 = 0
    m1 = 0
    u0 = 0
    u1 = 0
    E = 1.9*10**11
    y = twopBVP(fvec,m0,m1,L,N)
    
    y = np.delete(y,0)
    y = np.delete(y,-1)
    
    points = np.linspace(L/(N+1), L - L/(N+1), N)
    i = func_I(points,L)
    fvec = np.divide(y,i*E)
    
    u = twopBVP(fvec, u0, u1, L, N)
    return u
    