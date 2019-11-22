import numpy as np
from scipy import exp, sparse
from scipy.sparse.linalg import spsolve

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