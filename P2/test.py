from P2 import functions as f
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

# L = 10
#
# # Interior grid points
# N = 100
# alpha = 1
# beta = 2
#
# points = np.linspace(L/(N+1), L - L/(N+1), N)
# fvec = f.test_fsin(points)
# yexact = f.exact_sin(points, alpha, beta , L)
#
# y = f.twopBVP(fvec, alpha, beta, L, N)
#
# yexact = np.insert(yexact,0,alpha)
# yexact = np.append(yexact,beta)
#
# points = np.insert(points, 0, 0)
# points = np.append(points, L)
#
# plt.figure(1)
# plt.plot(points,y,'ob')
# plt.show()
# plt.plot(points,yexact,'or')
# plt.show()
# #K = np.arange(1,10)
# #K = 2**K
# #f.vshBvp(alpha, beta, L, K)
#
# #%% beam deflection section
# N = 999
# L = 10
# points = np.linspace(L/(N+1), L - L/(N+1), N)
#
# fvec = f.func_q(points)
#
# u = f.beamSolve(fvec, L, N)
#
# # Get beam midpoint
# print(np.median(u))
# points = np.insert(points, 0, 0)
# points = np.append(points, 10)
#
# plt.plot(points, u)
# plt.show()
#
# #%% section 2

alpha = 0
beta = 0
L = 1
N = 100



f.sturm_lv_vis()

