from P2 import functions as f
import numpy as np
import matplotlib.pyplot as plt
s = "hej"
L = 10

# Interior grid points
N = 100
alpha = 1
beta = 2

points = np.linspace(L/(N+1), L - L/(N+1), N)

fvec = f.test_f(points)

y = f.twopBVP(fvec, alpha, beta, L, N)

points = np.insert(points, 0, 0)
points = np.append(points, L)

plt.plot(points, y)
plt.show()