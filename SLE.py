import numpy as np
import PLUQ
import PLU


def solve_lin_eq(a, b):
    p, l, u, rank = PLU.decompose(a)

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
    x[rows-1, 0] = y[rows-1] / U[rows-1, cols-1]

    for i in range(2, rows+1):
        sum = 0
        for j in range(1, i):
            sum += x[rows-j, 0] * u[rows-i, cols-j]
        x[rows-i, 0] = (y[rows-i] - sum) / u[rows-i, rows-i]

    return x
    # check = PLU.matrix_multiplication(a, x)
    # print check


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
