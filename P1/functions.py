import numpy as np
from scipy.linalg import expm, norm, inv, det
import matplotlib.pyplot as plt
import time

# --- 1 ---

def eulerstep(A, u_old, h):
    # Basic eulerstep
    return u_old + h*np.dot(A, u_old)


def eulerint(A, y0, t0, tf, N):
    # Assume A matrix, y0 column vector, e.g np.array([[1],[1]])

    # Step size
    h = (tf-t0)/N

    u_old = y0
    rows = np.size(y0)

    approx = np.zeros((rows, N + 1))
    errors = np.zeros(N + 1)

    # First approx is y0, first error is 0.

    approx[:, 0] = y0[:,0]
    errors[0] = 0

    for i in range(1, N+1):
        u_new = eulerstep(A, u_old, h)

        approx[:, i] = u_new[:, 0]

        exact = np.dot(expm(A * (t0+h*i)), y0)

        errors[i] = norm(exact[:,0] - approx[:,i])

        u_old = u_new

    return approx, errors


def errorVSh(A, y0, t0, tf):
    # Visualization of error based on step size

    # Number of step sizes
    K = 10

    N = np.arange(K)
    N = np.power(2,N)

    h = (tf - t0)/N

    errors = np.zeros(K)

    for k in range(K):
        approx, err = eulerint(A, y0, t0, tf, N[k])
        errors[k] = err[-1]

    plt.loglog(h, errors)
    plt.show()


def timeErrorVSh(A, y0, t0, tf, N):
    # Plot the error (in log) as a function of time

    approx, err = eulerint(A, y0, t0, tf, N)

    t = np.linspace(t0, tf, N)

    plt.semilogy(t, err)

    plt.show()


def itimeErrorVSh(A, y0, t0, tf, N):
    # Plot the error (in log) as a function of time

    approx, err = ieulerint(A, y0, t0, tf, N)

    t = np.linspace(t0, tf, N)

    plt.semilogy(t, err)

    plt.show()


def ieulerstep(A, u_old, h):
    # Implicit Eulerstep
    A_size = np.alen(A)
    if det(np.eye(A_size) - h*A) == 0:
        raise Exception("A not invertible")
    return np.dot(inv(np.eye(A_size) - h*A), u_old)


def ieulerint(A, y0, t0, tf, N):
    # Approximates the solution to the given diff. eq
    # using implicit Eulerstep instead of explicit

    # Step size
    h = (tf-t0)/N

    u_old = y0
    rows = np.size(y0, 0)

    approx = np.zeros((rows, N + 1))
    errors = np.zeros(N + 1)

    # First approx is y0, first error is 0.

    approx[:, 0] = y0[:, 0]
    errors[0] = 0

    for i in range(1,N + 1):
        u_new = ieulerstep(A, u_old, h)

        # Size mismatch without .ravel()

        # approx[:, i] = u_new.ravel()
        approx[:, i] = u_new[:, 0]

        exact = np.dot(expm(A * (t0+h*i)), y0)

        errors[i] = norm(exact[:,0] - approx[:,i])

        u_old = u_new

    return approx, errors


def ierrorVSh(A, y0, t0, tf):
    # Visualization of error based on step size
    # for implicit method

    # Number of step sizes
    K = 10

    N = np.arange(K)
    N = np.power(2,N)

    h = (tf - t0)/N

    errors = np.zeros(K)

    for k in range(K):
        approx, err = ieulerint(A, y0, t0, tf, N[k])
        errors[k] = err[-1]

    print(h)
    print(errors)
    plt.loglog(h, errors)
    plt.show()

# --- 2 ---

def test_f(t, y, mu = 0):
    # linear test equation
    lam = -1
    return lam*y


def lotka_vol(t, y_v, mu = 0):
    # y column vector of size 2

    # Define system parameters

    a = 3
    b = 9
    c = 15
    d = 15

    x = y_v[0, 0]
    y = y_v[1, 0]

    xd = a * x - b * x * y
    yd = c * x * y - d * y

    return np.array([[xd], [yd]])


def van_pol(t, y_v, mu = 100):
    y1 = y_v[0, 0]
    y2 = y_v[1, 0]

    y1d = y2
    y2d = mu * (1 - y1 * y1) * y2 - y1

    return np.array([[y1d], [y2d]])


def rkstep(f, y_old, t_old, h, A, b):
    # Explicit Runge-Kutta method of order defined by A
    # f is here the given diff. eq. on the form y´ = f(t, y)

    n_stages = np.size(b)

    c = np.array([np.sum(e) for e in A])

    # Store n_derivs stage derivatives for size(y_old) function values
    Y_stages = np.zeros((np.size(y_old), n_stages))
    # print('ystages')
    # print(np.shape(Y_stages))

    for i in range(n_stages):

        # Calculate next y value

        y_temp = np.zeros(np.size(y_old))

        for j in range(np.size(y_old)):
            y_temp = y_temp + h * np.dot(A[i, :], np.transpose(Y_stages[j, :]))

        # y_temp += y_old

        y_temp = np.add(y_temp, y_old)

        Y_stages[:, i] = f(t_old + c[i] * h, y_temp)[:, 0]

    y_next = y_old

    # Add weighted stage derivatives to y_next

    for k in range(n_stages):
        y_to_add =  h * b[k] * np.reshape(Y_stages[:, k], (np.size(y_old), 1))

        y_next = np.add(y_next, y_to_add)

    # print('Y_stages')
    # print(Y_stages.shape)
    # print(Y_stages)

    return y_next, Y_stages

def rk4step_2(f, y_old, t_old, h):

    # 4-step utan att använta den generella rkstep
    # Kanske snabbare?

    A = np.array([[0, 0, 0, 0],
                  [1 / 2, 0, 0, 0],
                  [0, 1 / 2, 0, 0],
                  [0, 0, 1, 0]])

    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    # c = np.array([np.sum(e) for e in A])
    c = np.array([0, 0.5, 0.5, 1])

    n_stages = np.size(b)
    n_vars = np.size(y_old)

    Y_stages = np.zeros((n_vars, n_stages))




def rk4step(f, y_old, t_old, h):
    # Explicit Runge-Kutta method of order 4
    # f is here the given diff. eq. on the form y´ = f(t, y)
    # Returns the estimated value as well as the
    # stage derivatives.

    A = np.array([[0,   0, 0, 0],
                  [1/2, 0, 0, 0],
                  [0, 1/2, 0, 0],
                  [0,   0, 1, 0]])

    b = np.array([1/6, 1/3, 1/3, 1/6])

    # c = np.array([[0], [1/2], [1/2], [1]])

    return rkstep(f, y_old, t_old, h, A, b)


def rk4int(f, y0, t0, tf, N):
    # Step size
    h = (tf-t0)/N

    y_old = y0
    t_old = t0

    rows = np.size(y0)

    approx = np.zeros((rows, N + 1))
    errors = np.zeros(N + 1)

    # First approx is y0, first error is 0.

    approx[:, 0] = y0[:, 0]
    errors[0] = 0

    print(approx)

    for i in range(1, N+1):
        output = rk4step(f, y_old, t_old, h)

        # We only want the new y value, disregard Y_stage
        y_new = output[0]
        approx[:, i] = y_new[:, 0]

        # Solution to our given function f

        # for exponential matrix:
        # exact = np.dot(expm(f(0, 1) * (t0+h*i)), y0)

        # for single variable
        exact = y0 * np.exp(f(0,1) * (t0+h*i))

        errors[i] = norm(exact[:, 0] - approx[:, i])

        y_old = y_new

    return approx, errors


def rk3step(f, y_old, t_old, h):
    # Returns the esimated y_next value for the diff. eq. defined by
    # f, as well as the computed stage derivatives used to obtain it.

    A = np.array([[0, 0, 0],
                  [1/2, 0, 0],
                  [-1, 2, 0]])

    b = np.array([1 / 6, 2 / 3, 1 / 6])

    return rkstep(f, y_old, t_old, h, A, b)


def rk34step(f, y_old, t_old, h):
    # Used 3 and 4 step RK methods to estimate the next y value and
    # the error.

    # Gives numerical inaccuracy in y_new, but takes faster
    # (Even though we do more computations ??).

    y_new, Y_s4 = rk4step(f, y_old, t_old, h)

    _, Z_s3 = rk3step(f, y_old, t_old, h)

    # Estimate error using rk3, index starts at 0
    err = h/6 * (2 * Y_s4[:, 1] + Z_s3[:, 2] - 2 * Y_s4[:, 2] - Y_s4[:, 3])

    return y_new, norm(err)


def rk34embedstep(f, y_old, t_old, h):
    # Embedded rk34 method with minimized unnecessary computed derivatives.
    # Only calculated the necessary stage derivatives for error estimation
    # from the third order method.

    y_new, Y_s4 = rk4step(f, y_old, t_old, h)

    A3 = np.array([[0, 0, 0],
                  [1 / 2, 0, 0],
                  [-1, 2, 0]])

    # b3 = np.array([1 / 6, 2 / 3, 1 / 6])
    c3 = np.array([np.sum(e) for e in A3])

    # Third stage derivative for RK3

    Y_s4_cut = Y_s4[:,:-1]

    Z_s3_3 = f(t_old + c3[2] * h, y_old + h * np.dot(A3[2, :], np.transpose(Y_s4_cut)))

    # Estimate error using rk3, index starts at 0
    err = h / 6 * (2 * Y_s4[:, 1] + Z_s3_3 - 2 * Y_s4[:, 2] - Y_s4[:, 3])

    return y_new, norm(err)


def newstep(tol, err, errold, hold, k):
    return np.power(tol/err, 2/(3*k)) * np.power(tol/errold, -1/(3*k)) * hold


def adaptive_rk34(f, y0, t0, tf, tol):
    # Solves f using rk34 while keeping the error estimate equal to tol

    start_time = time.time()

    h0 = np.abs(tf - t0) * np.power(tol, 0.25) / 100
    h0 = h0 / (1 + norm(f(0, y0)))

    k = 4

    h_old = h0
    t_old = t0
    y_old = y0

    # Empty array for times
    times = np.zeros(100)
    times[0] = t0
    approx = y0

    err_old = tol

    n_it = 0
    i = 1

    while(t_old < tf):
        y_new, err = rk34embedstep(f, y_old, t_old, h_old)

        # Appends y_new to our currents approximations
        approx = np.hstack([approx, y_new])

        t_old = t_old + h_old

        # Slow?
        # times = np.append(times, t_old)

        curr_size = np.size(times)
        if i == curr_size:
            # Larger time vector needed

            times_new = np.zeros(2 * curr_size)
            times_new[:curr_size] = times[:]
            times = times_new

        times[i] = t_old

        h_new = newstep(tol, err, err_old, h_old, k)

        h_old = h_new
        err_old = err
        y_old = y_new

        i = i + 1

        n_it = n_it + 1

        # Print progess
        print(t_old / tf)

    # We have now stepped past tf, get final value at tf

    # Removes all unneccesary zeros
    times = times[:i]

    t_penult = times[np.size(times) - 2]
    h_final = tf - t_penult
    times[np.size(times) - 1] = tf
    y_new, _ = rk34step(f, y_old, t_penult, h_final)

    times = np.reshape(times, (1, np.size(times)))

    end_time = time.time()

    print('number of iterations: ' + str(n_it))

    run_time = end_time - start_time

    print('runtime: {}'.format(run_time))

    return times, approx


