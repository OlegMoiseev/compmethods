import numpy as np


def matrix_multiplication(a, b):
    rows, _ = a.shape
    _, cols = b.shape
    mat = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            for k in range(rows):
                mat[i, j] += a[i, k] * b[k, j]
    return mat


def matrix_determinant(u):
    rows, cols = u.shape
    det = 1.
    for i in range(rows):
        det *= u[i, i]
    if rows % 2 and rows != 1:
        det *= -1
    return det


def swap_rows(m, s, f):
    tmp = np.copy(m[f, :])
    m[f, :] = np.copy(m[s, :])
    m[s, :] = tmp


def swap_cols(m, s, f):
    tmp = np.copy(m[:, f])
    m[:, f] = np.copy(m[:, s])
    m[:, s] = tmp


def decompose(a_orig):
    a = np.copy(a_orig)
    rows, cols = a.shape
    p = np.eye(rows, cols)
    rank = rows

    for k in range(rows):
        pivot_value = 0

        for i in range(k, rows):
            if abs(a[i, k]) > pivot_value:
                pivot_value = abs(a[i, k])
                row_with_max_elem = i

        if pivot_value < 10e-16:
            rank -= 1
            continue

        swap_rows(a, k, row_with_max_elem)
        swap_rows(p, k, row_with_max_elem)

        for i in range(k + 1, rows):
            a[i, k] /= a[k, k]
            for j in range(k + 1, rows):
                a[i, j] -= a[i, k] * a[k, j]

    lower = np.zeros((rows, cols))
    upper = np.zeros((rows, cols))

    for i in range(rows):
        lower[i, i] = 1.
        for j in range(cols):
            if i > j:
                lower[i, j] = a[i, j]
            else:
                upper[i, j] = a[i, j]

    return p, lower, upper, rank


def check(a, p, l, u):
    print "Original matrix:"
    print a
    print "P^(-1)*L*U:"
    plu_mat = matrix_multiplication(p.T, matrix_multiplication(l, u))
    print plu_mat
    print "Matrices are equivalent:", np.allclose(a, plu_mat, atol=10e-16)
