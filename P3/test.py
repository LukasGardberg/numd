import numpy as np
from scipy.linalg import expm, norm, inv, det
import matplotlib.pyplot as plt
import P3.functions as f
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

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