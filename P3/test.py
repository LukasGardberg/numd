import numpy as np
from scipy.linalg import expm, norm, inv, det 
import matplotlib.pyplot as plt
import P3.functions as f
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


#%% del 1
N = 20
M = 200
L = 1
T = 1

xx = np.linspace(0, L, N + 2)
tt = np.linspace(0, T, M + 1)

fvec = f.startVal(xx)
fvec = np.delete(fvec, -1)
fvec = np.delete(fvec, 0)

sol = f.diffSolve(N, M, L, T, fvec)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx, tt = np.meshgrid(xx, tt)

ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()

#%% del 2
a = -0.29
N = 200
M = 400


xx = np.linspace(0, 1, N + 1)
tt = np.linspace(0, 5, M + 1)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')

sol = f.advecSolve(N,M,f.startG,a)
xx, tt = np.meshgrid(xx,tt)

ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()

err = np.zeros(M+1)
t = np.linspace(0, 5, M + 1)
for i in range(M+1):
    err[i] = 1/np.sqrt(M+1) * norm(sol[i,:])

fig2 = plt.figure(2)
plt.plot(t,err)
plt.show()




