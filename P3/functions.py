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


def LaxWenStep(uold, amu):
    size = np.size(uold)
    sub = np.ones(size - 1) * (amu/2) * (1 + amu)
    mid = np.ones(size) * (1 - amu**2)
    sup = np.ones(size - 1) * (-amu/2) * (1 - amu)

    bot = np.ones(1) * (amu / 2) * (amu - 1)
    top = np.ones(1) * (amu / 2) * (amu + 1)

    diagonals = [bot, sub, mid, sup, top]

    Tdx = sparse.diags(diagonals, [-(size - 1), -1, 0, 1, size - 1])

    return Tdx.dot(uold)


def diffSolve(N, M, L, T, fvec):
    # Solves diffusion eq

    dx = L / (N + 1)
    dt = T / M

    # print(dt/(dx*dx))

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


def startG(x):
    return np.exp(-100*(x-0.5)**2)


def startF(x):
    return -x**2 + x


def advecSolve(N, M, g, a):
    T = 5

    dx = 1 / N
    dt = T / M
    amu = a * (dt / dx)

    xx = np.linspace(0, 1, N + 1)
    tt = np.linspace(0, T, M + 1)

    uold = g(xx)
    uold = np.delete(uold, -1)

    sol = np.zeros((M + 1, N))

    sol[0, :] = uold

    for i in range(1, M+1):
        unew = LaxWenStep(uold, amu)

        sol[i, :] = unew

        uold = unew

    # Add first column as last, periodic boundry
    sol = np.hstack((sol, np.reshape(sol[:, 0], (M + 1, 1))))
    print(amu)
    return sol


def convdifstep(uold, Tdx, Sdx, a, d, dt):

    diff = d * Tdx - a * Sdx

    unit = sparse.diags([np.ones(len(uold))], [0])

    lhs = unit - (dt/2) * diff
    rhs = unit + (dt/2) * diff

    return spsolve(lhs, rhs.dot(uold))


def convdifsolve(N, M, g, a, d):
    # Solver for convection-diffusion equation
    # Assume x \in [0, 1], t \in [0, 1]

    dx = 1 / N
    dt = 1 / M

    # Create Tdx (diffusion)
    subp = np.ones(N - 1)
    mid = np.ones(N) * (-2)

    bottop = np.ones(1)

    diagsT = [bottop, subp, mid, subp, bottop]

    Tdx = 1 / (dx * dx) * sparse.diags(diagsT, [-(N-1), -1, 0, 1, (N-1)])

    # Create Sdx (advection)
    sub = -np.ones(N - 1)
    sup = np.ones(N - 1)

    bot = np.ones(1)
    top = -np.ones(1)

    diagsS = [bot, sub, sup, top]

    Sdx = 1 / (2 * dx) * sparse.diags(diagsS, [-(N - 1), -1, 1, (N - 1)])

    # Find startvalues
    xx = np.linspace(0, 1, N + 1)
    uold = g(xx)
    uold = np.delete(uold, -1)

    # Create solution matrix
    sol = np.zeros((M + 1, N))

    sol[0, :] = uold

    for i in range(1, M + 1):
        unew = convdifstep(uold, Tdx, Sdx, a, d, dt)

        sol[i, :] = unew

        uold = unew

    # Add first column as last, periodic boundry
    sol = np.hstack((sol, np.reshape(sol[:, 0], (M + 1, 1))))

    print(f"Mesh Pelcet = {abs(a/d) * dx}")

    return sol
