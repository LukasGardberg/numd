from P1 import functions as f
import matplotlib.pyplot as plt
import numpy as np

# A = np.array([[-1, 100], [0, -30]])
# t0 = 0
# tf = 10
# y0 = np.array([[1],[1]])
# N = 10

# approx, errors = f.eulerint(A, y0, t0, tf, N)

# f.errorVSh(A, y0, t0, tf)
# f.ierrorVSh(A, y0, t0, tf)

# f.timeErrorVSh(A, y0, t0, tf, 100)
# f.itimeErrorVSh(A, y0, t0, tf, 100)

# t0 = 0
# tf = 10
# y0 = np.array([[1]])
# tol = 1e-6

# approx, errors = f.rk4int(f.test_f, y0, t0, tf, N)

# times = np.arange(0, 11)

# plt.loglog(times, errors)
# plt.show()

# f.rk34step(f.test_f, y0, t0, 1)

# times, approx = f.adaptive_rk34(f.test_f, y0, t0, tf, tol)
# plt.plot(times[0, :], approx[0, :])
# plt.show()
#
# print(np.shape(times))
# print(np.shape(approx))
#
# print(approx)

# RK4step -------------------

# tol = 0.001
#
# t, y = f.adaptive_rk34(f.test_f, y0, t0, tf, tol)
#
# t_exact = np.arange(t0, tf, 0.1)
#
# plt.plot(t, y, 'ro')
# plt.plot(t_exact, np.exp(-t_exact), '-')
#
# plt.show()

# Lotka Volterra ------------

t0 = 0
tf = 5
y0 = np.array([[1], [1]])
tol = 10**-4

times, approx = f.adaptive_rk34(f.lotka_vol, y0, t0, tf, tol)

plt.figure(1)

plt.plot(times[0, :], approx[0, :])
plt.plot(times[0, :], approx[1, :])

plt.show()

plt.figure(2)
plt.plot(approx[0, :], approx[1, :])
plt.show()

# Van der Pol ---------------

# mu = 100
# y0 = np.array([[2], [0]])
# t0 = 0
# tf = 2*mu
# tol = 10**-6
#
# times, approx = f.adaptive_rk34(f.van_pol, y0, t0, tf, tol)
#
# plt.figure(1)
# plt.plot(times[0, :], approx[1, :])
# plt.show()
#
#
# plt.figure(2)
# plt.plot(approx[0, :], approx[1, :])
# plt.show()

# Different mu _--------------

# mu = 100
# y0 = np.array([[2], [0]])
# t0 = 0
# tf = 0.7*mu
# tol = 10**-6
#
# mu = np.array([10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680, 1000])
#
# steps = np.zeros(np.size(mu))
# i = 0
# for m in mu:
#     times, approx = f.adaptive_rk34(f.van_pol, y0, t0, tf, tol, mu)
#     steps[i] = np.size(times) - 1
#     i = i + 1
#
# print(steps)
# plt.loglog(steps, mu)
# plt.show()

## -------------
# times = np.zeros(5)
#
# i = 1
# while(i < 19):
#     curr_size = np.size(times)
#     if(i == curr_size):
#
#         times_new = np.zeros(2 * curr_size)
#         times_new[:curr_size] = times[:]
#         times = times_new
#
#     times[i] = 2
#
#     i = i + 1
#
# print(times)
#
# times = times[:i]