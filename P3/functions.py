import numpy as np
from scipy.linalg import expm, norm, inv, det
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import scipy as sp
def eulerstep(Tdx, yold, dt):
    # yold np array
    return yold + dt * Tdx.dot(yold)

def TRstep(Tdx, yold, dt):
    a = yold + dt/2 * Tdx.dot(yold)

    mid = np.ones(Tdx.shape[0])
    b = sparse.diags([mid], [0]) - Tdx * dt/2

    return spsolve(b.tocsc(), a)

def LaxWenStep(yold, dt, dx):
    mu = dt / dx
    
    pass

def diffSolve(N, M, L, T, fvec):
    # Solves diffusion eq

    dx = L / (N + 1)
    dt = T / M

    print(dt/(dx*dx))

    subp = np.ones(N - 1)
    mid = np.ones(N) * (-2)

    diagonals = [subp, mid, subp]

    Tdx = 1/(dx * dx) * sparse.diags(diagonals, [-1, 0, 1])

    sol = np.zeros((M + 1, N))
    sol[0, :] = fvec

    # yold = eulerstep(Tdx, fvec, dt)
    yold = TRstep(Tdx, fvec, dt)

    sol[1, :] = yold

    for i in range(2, M):
        # ynew = eulerstep(Tdx, yold, dt)
        ynew = TRstep(Tdx, yold, dt)

        sol[i, :] = ynew
        yold = ynew

    boundry = np.zeros((M + 1, 1))
    sol = np.hstack([sol, boundry])
    sol = np.hstack([boundry, sol])

    return sol

def startVal(x):
    return np.exp(-(x-0.5)**2) - np.exp(-0.25)