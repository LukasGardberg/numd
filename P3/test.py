import numpy as np
from scipy.linalg import expm, norm, inv, det 
import matplotlib.pyplot as plt
import P3.functions as f
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.ticker as ticker

 #%% del 1
N = 20
M1 = 30
M2 = 124
L = 1
T = 0.15

xx = np.linspace(0, L, N + 2)
tt = np.linspace(0, T, M1 + 1)

fvec = f.startVal(xx)
fvec = np.delete(fvec, -1)
fvec = np.delete(fvec, 0)

sol = f.diffSolve(N, M1, L, T, fvec)
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx, tt = np.meshgrid(xx, tt)
 
 
ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot CFl=2.21')
ax.set_xlabel('t')
 ax.set_ylabel('x')
 
#ax2 = fig.add_subplot(122, projection='3d')
tt2 = np.linspace(0, T, M2 + 1)
#xx2 = np.linspace(0, L, N + 2)
#sol2 = f.diffSolve(N, M2, L, T, fvec)
#xx2, tt2 = np.meshgrid(xx2, tt2)
 
#ax2.plot_surface(tt2, xx2, sol2, cmap='viridis', edgecolor='none')
#ax2.set_title('Surface plot CFl>0.5')
#ax2.set_xlabel('t')
#ax2.set_ylabel('x')
 
 
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_zticklabels([])
#ax2.set_xticklabels([])
#ax2.set_yticklabels([])
plt.show()
 
 

#%% del 2
a1 = 1.2
a2= 0.04
N = 100
M = 600
N2 = 1000
xx = np.linspace(0, 1, N + 1) 
tt = np.linspace(0, 5, M + 1)
xx2= np.linspace(0,1,N2+1)
# fig1 = plt.figure()
# ax = fig1.add_subplot(121, projection='3d')
fig, (ax1,ax2) = plt.subplots(1,2)
 
#
sol = f.advecSolve(N, M, f.startG, a1)
sol2= f.advecSolve(N2,M,f.startG,a2)
 

#ax1.xaxis.set_ticks(np.arange(0,1))
#ax1.yaxis.set_ticks(np.arange(0,5))
#ax1.colorbar()
#ax2.set(xlim=(0,1),ylim=(0,5))
ax1.imshow(np.transpose(sol),extent=(0,5,0,1),aspect='auto')
ax2.imshow(np.transpose(sol2),extent= (0,5,0,1),aspect='auto')
#ax2.set(xlim(0,1),ylim(0,5))
ax1.set_xlabel('t')
ax2.set_xlabel('t')
ax1.set_ylabel('x')

ax1.set_title('Advection equation')
ax2.set_title('Advection equation')
plt.show()
# X, T = np.meshgrid(xx, tt)
#
# ax.plot_surface(T, X, sol, cmap='viridis', edgecolor='none')
# ax.set_title('Surface plot')
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# 
# ax2 = fig1.add_subplot(122,projection='3d')
# xx2 = np.linspace(0, 1, N2 + 1)
#  
# sol2 = f.advecSolve(N2, M, f.startG, a)
# 
# X2, T2 = np.meshgrid(xx2, tt)
#
## ax2.plot_surface(T2, X2, sol2, cmap='viridis', edgecolor='none')
## ax2.set_title('Surface plot')
## ax2.set_xlabel('t')
## ax2.set_ylabel('x')
## ax2.set_xticklabels([])
## ax2.set_yticklabels([])
# plt.figure(3)
# plt.imshow(sol2,cmap='hot')
# plt.show()
# xx = np.linspace(0, 1, N + 1)
# tt = np.linspace(0, 5, M + 1)
# 
# 
# 
# plt.show()
fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)

rms1 = np.zeros(M + 1)
rms2 = np.zeros(M+1)
t = np.linspace(0, 5, M + 1)
for i in range(M+1):
    rms1[i] = 1 / np.sqrt(N) * norm(sol[i, :])
    rms2[i] = 1/np.sqrt(N2)*norm(sol2[i,:])
 
ax1.plot(t,rms1)
#ax2.plot(t,rms2)
ax1.set_xlabel("t")
ax2.set_xlabel("t")
ax1.set_ylabel("RMS-norm")
ax1.set_title("RMS norm vs t, CFL=1")
ax2.set_title("RMS norm vs t, CFL=0.9")
plt.show()

#%% Del 3

# Points in space
N = 200
# Points in time
M = 800

a = 1.5
d = -0.1

xx = np.linspace(0, 1, N + 1)
tt = np.linspace(0, 5, M + 1)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')

sol = f.convdifsolve(N, M, f.startG, a, d)
xx, tt = np.meshgrid(xx, tt)

ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.set_xlabel('t')
ax.set_ylabel('x')

plt.show()

#%% Part 4

# Points in space
N = 250
# Points in time
M = 1500

d = 0.001

xx = np.linspace(0, 1, N + 1)
tt = np.linspace(0, 1, M + 1)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')

sol = f.visBurgSolve(N, M, f.startG, d)
xx, tt = np.meshgrid(xx, tt)

ax.plot_surface(tt, xx, sol, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()