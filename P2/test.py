from P2 import functions as f
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


#%% task 1.1
L = 10
 #
 # # Interior grid points
N = 100
alpha = 3
beta = 1
points = np.linspace(L/(N+1), L - L/(N+1), N)
fvec = f.test_fsin(points)
yexact = f.exact_sin(points, alpha, beta , L)

y = f.twopBVP(fvec, alpha, beta, L, N)

yexact = np.insert(yexact,0,alpha)
yexact = np.append(yexact,beta)

points = np.insert(points, 0, 0)
points = np.append(points, L)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(points,y,'ob',label = "Numerical solution")
plt.show()
ax.plot(points,yexact,'r',label ="Exact solution")
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("y")

plt.title("Exact and numerical solution plotted together")

K = np.arange(1,10)
K = 2**K
plt.figure(2)
f.vshBvp(alpha, beta, L, K)

#%% task 1.2 beam deflection

N = 999
L = 10
points = np.linspace(L/(N+1), L - L/(N+1), N)
 
fvec = f.func_q(points)

u = f.beamSolve(fvec, L, N)

 # Get beam midpoint
print(u[501])
points = np.insert(points, 0, 0)
points = np.append(points, 10)
plt.grid()
plt.xlabel("x")
plt.ylabel("Delfection in mm")
plt.title("Deflection vs x")
plt.plot(points, u)
plt.show()

#%% section 2 sturm lioville

alpha = 0
beta = 0
L = 1
N = 100

f.sturm_lv_vis()

#%% schrödinger 
L = 1
N = 1000

a = f.shrod_solve(L, N, f.potential_v2, 7)
print(a) 