import numpy as np
from scipy import exp, sparse
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


def exact_f(x, alpha, beta, L):
    return exp(-x) + alpha - 1 + (x/L) * (beta - alpha + 1 - exp(-L))


def vshBvp(alpha, beta, length, N):
    errors = np.array([])
    step_sizes = np.array([])

    for k in N:
        print(length)
        print(k)
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