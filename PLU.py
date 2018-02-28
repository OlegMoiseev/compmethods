import numpy as np


def matrix_multiplication(a, b):
    rows, _ = a.shape
    _, cols = b.shape
    mat = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(rows):
            for k in range(rows):
                mat[i, j] += a[i, k] * b[k, j]
    return mat


def matrix_determinant(u):
    det = 1.
    for i in range(3):
        det *= u[i, i]
    return det


def swap_rows(m, s, f):
    tmp = np.copy(m[f, :])
    m[f, :] = np.copy(m[s, :])
    m[s, :] = tmp


def plu(a):
    rows, cols = a.shape
    p = np.eye(rows, cols)
    for k in range(rows):
        pivot_value = 0
        for i in range(k, rows):
            if abs(a[i, k]) > pivot_value:
                pivot_value = abs(a[i, k])
                row_with_max_elem = i

        if pivot_value == 0:
            raise Exception("Degenerate matrix")

        swap_rows(a, k, row_with_max_elem)
        swap_rows(p, k, row_with_max_elem)

        for i in range(k + 1, rows):
            a[i, k] /= a[k, k]
            for j in range(k + 1, rows):
                a[i, j] -= a[i, k] * a[k, j]

    l = np.zeros((rows, cols))
    u = np.zeros((rows, cols))

    for i in range(rows):
        l[i, i] = 1.
        for j in range(cols):
            if i > j:
                l[i, j] = a[i, j]
            else:
                u[i, j] = a[i, j]

    return p, l, u


def check_plu(p, l, u):
    return matrix_multiplication(p, matrix_multiplication(l, u))


A = np.array([[2., 7., -6.],
              [8., 2., 1.],
              [7., 4., 2.]])

P, L, U = plu(A)

print check_plu(P, L, U)

