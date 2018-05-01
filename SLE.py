import numpy as np
import PLUQ
import PLU


def solve_lin_eq(a, b):
    p, l, u, q, rank, t = PLUQ.decompose(a)

    rows, cols = a.shape
    y = np.zeros(rows)

    b = PLU.matrix_multiplication(p, b)

    y[0] = b[0]

    for i in range(1, rows):
        sum = 0
        for j in range(i):
            sum += y[j] * l[i, j]
        y[i] = b[i] - sum

    x = np.zeros((rows, 1))

    for j in range(rows-1, 0, -1):
        if np.allclose(u[j], x, atol=10e-9) and abs(y[j]) < 10e-9:
            rows -= 1
            cols -= 1
        elif np.allclose(u[j], x, atol=10e-9) and not abs(y[j]) < 10e-9:
            print("Isn't consistent!")
            return x

    x[rows-1, 0] = y[rows-1] / u[rows-1, cols-1]

    for i in range(2, rows+1):
        sum = 0
        for j in range(1, i):
            sum += x[rows-j, 0] * u[rows-i, cols-j]
        x[rows-i, 0] = (y[rows-i] - sum) / u[rows-i, rows-i]
    x = PLU.matrix_multiplication(q, x)
    return x


def check(a, x, b):
    dim, _ = a.shape
    ax = PLU.matrix_multiplication(a, x)
    res = np.zeros((dim, 1))
    z = np.zeros((dim, 1))
    for i in range(dim):
        res[i] = ax[i, 0] - b[i, 0]

    return np.allclose(z, res, atol=10e-9)


def solve_lin_eq_wd(p, l, u, q, b):
    rows, cols = p.shape

    y = np.zeros(rows)

    b = PLU.matrix_multiplication(p, b)

    y[0] = b[0]

    for i in range(1, rows):
        sum = 0
        for j in range(i):
            sum += y[j] * l[i, j]
        y[i] = b[i] - sum

    x = np.zeros((rows, 1))

    for j in range(rows-1, 0, -1):
        if np.allclose(u[j], x, atol=10e-9) and abs(y[j]) < 10e-9:
            rows -= 1
            cols -= 1
        elif np.allclose(u[j], x, atol=10e-9) and not abs(y[j]) < 10e-9:
            print("Isn't consistent!")
            return x

    x[rows-1, 0] = y[rows-1] / u[rows-1, cols-1]

    for i in range(2, rows+1):
        sum = 0
        for j in range(1, i):
            sum += x[rows-j, 0] * u[rows-i, cols-j]
        x[rows-i, 0] = (y[rows-i] - sum) / u[rows-i, rows-i]
    x = PLU.matrix_multiplication(q, x)
    return x

'''A = np.array([[2., 7., 6.],
              [8., 2., 1.],
              [7., 4., 2.]])

B = np.array([[2., 7., -6., 5., -7.],
              [8., 2., 1., -56., 2.],
              [7., 4., 2., 9., 3.],
              [-1., 4., -6., -9., 3.],
              [7., 4., 2., 9., 45.]])

C = np.array([[2., 7.],
              [8., 2.]])

D = np.array([[2., 7., -6., 5., -7., 4.],
              [8., 2., 1., -56., 2., 9.],
              [21., 4., 45., 9., 3., 2.],
              [-1., 4., -6., -9., 3., -4.],
              [7., 4., 2., 9., 45., -7],
              [-12., 34., -26., -19., 23., -54.]])

solve_lin_eq(D, np.array([[1], [2], [3], [4], [5], [2]]))'''
