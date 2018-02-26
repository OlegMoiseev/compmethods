import numpy as np


def matrix_multiplication(a, b):
    c = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])

    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j] += a[i][k] * b[k][j]
    return c


A = np.array([[10., -7., 0.],
              [-3., 6., 2.],
              [5., -1., 5.]])

L = np.array([[A[0][0],      0.,       0.],
              [A[1][0], A[1][1],       0.],
              [A[2][0], A[0][1], A[2][2]]])

U = A

"""for j in range(3):
    U[0][j] = A[0][j]

for j in range(1, 3):
    L[j][0] = A[j][0]/U[0][0]

for i in range(1, 3):
    for j in range(i, 3):
        s = 0.
        for k in range(i):
            s += L[i][k] * U[k][j]
        U[i][j] = A[i][j] - s

    for j in range(i, 3):
        s = 0
        for k in range(i):
            s += L[i][k] * U[k][j]
        L[j][i] = (A[j][i] - s)/U[i][i]
"""
for i in range(3):
    for j in range(3):
        U[0, i] = A[0, i]
        L[i, 0] = A[i, 0] / U[0, 0]
        s = 0.
        for k in range(i):
            s += L[i, k] * U[k, j]

        U[i, j] = A[i, j] - s
        if i > j:
            L[j, i] = 0
        else:
            s = 0.
            for k in range(i):
                s += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - s) / U[i, i]

print matrix_multiplication(L, U)
