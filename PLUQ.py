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
    rows, cols = u.shape
    det = 1.
    for i in range(rows):
        det *= u[i, i]
    if rows % 2:
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


def pluq(a_orig):
    a = np.copy(a_orig)
    rows, cols = a.shape
    p = np.eye(rows, cols)
    q = np.eye(rows, cols)

    for k in range(rows):
        pivot_value = 0

        for i in range(k, rows):
            for j in range(k, cols):
                if abs(a[i, j]) > pivot_value:
                    pivot_value = abs(a[i, j])
                    row_with_max_elem = i
                    col_with_max_elem = j

        if pivot_value < 10e-16:
            raise Exception("Degenerate matrix")

        swap_rows(a, k, row_with_max_elem)
        swap_cols(a, k, col_with_max_elem)

        swap_cols(p, k, row_with_max_elem)
        swap_rows(q, k, col_with_max_elem)

        for i in range(k + 1, rows):
            a[i, k] /= a[k, k]
            for j in range(k + 1, cols):
                a[i, j] -= a[i, k] * a[k, j]

    lower = np.eye(rows, cols)
    upper = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if i > j:
                lower[i, j] = a[i, j]
            else:
                upper[i, j] = a[i, j]

    return p, lower, upper, q


def check_pluq(a, p, l, u, q):
    print "Original matrix:"
    print a
    print "P*L*U*Q:"
    plu_mat = matrix_multiplication(matrix_multiplication(p, matrix_multiplication(l, u)), q)
    print plu_mat

    print "Matrices are equivalent:", np.allclose(a, plu_mat, atol=10e-16)


A = np.array([[2., 7., 6.],
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
              [7., 4., 2., 9., 3., 2.],
              [-1., 4., -6., -9., 3., -4.],
              [7., 4., 2., 9., 45., -7],
              [-12., 34., -26., -19., 23., -54.]])
try:

    P, L, U, Q = pluq(D)
    check_pluq(D, P, L, U, Q)
    # print "Det:", matrix_determinant(U)

except Exception as e:
    print e
