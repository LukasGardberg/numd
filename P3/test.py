import numpy as np
from scipy.linalg import expm, norm, inv, det 
import matplotlib.pyplot as plt
import P3.functions as f
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


 #%% del 1
 N = 20
 M = 87
 L = 1
 T = 0.1

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
 ax.set_xlabel('X')
 plt.show()

 #%% del 2
 a = 0.096
 N = 100
 M = 60


 xx = np.linspace(0, 1, N + 1)
 tt = np.linspace(0, 5, M + 1)

 fig1 = plt.figure(1)
 ax = fig1.add_subplot(111, projection='3d')

 sol = f.advecSolve(N, M, f.startG, a)
 xx, tt = np.meshgrid(xx, tt)

 ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
 ax.set_title('Surface plot')
 plt.show()

 rms = np.zeros(M + 1)
 t = np.linspace(0, 5, M + 1)
 for i in range(M+1):
     rms[i] = 1 / np.sqrt(N) * norm(sol[i, :])

 fig2 = plt.figure(2)
 plt.plot(t, rms)
 plt.show()

 #%% Del 3

 # Points in space
 N = 100
 # Points in time
 M = 1000

 a = 1
 d = 0.1

 xx = np.linspace(0, 1, N + 1)
 tt = np.linspace(0, 5, M + 1)

 fig1 = plt.figure(1)
 ax = fig1.add_subplot(111, projection='3d')

 sol = f.convdifsolve(N, M, f.startG, a, d)
 xx, tt = np.meshgrid(xx, tt)

 ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
 ax.set_title('Surface plot')
 plt.show()

#%% Part 4

# Points in space
N = 50
# Points in time
M = 1000

d = 0.1

xx = np.linspace(0, 1, N + 1)
tt = np.linspace(0, 1, M + 1)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')

sol = f.visBurgSolve(N, M, f.startG, d)
xx, tt = np.meshgrid(xx, tt)

ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()