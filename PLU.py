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


def matrix_determinant(p, l, u):
    ml = 1.
    mu = 1.
    rows, cols = p.shape

    for i in range(rows):
        ml *= l[i, i]
        mu *= u[i, i]

    det = ml * mu
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


def plu(a_orig):
    a = np.copy(a_orig)
    rows, cols = a.shape
    p = np.eye(rows, cols)
    for k in range(rows):
        pivot_value = 0
        for i in range(k, rows):
            if abs(a[i, k]) > pivot_value:
                pivot_value = abs(a[i, k])
                row_with_max_elem = i

        if pivot_value < 10e-16:
            raise Exception("Degenerate matrix")

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

    return p, lower, upper


def check_plu(a, p, l, u):
    print "Original matrix:"
    print a
    print "P*L*U:"
    plu_mat = matrix_multiplication(p, matrix_multiplication(l, u))
    print plu_mat
    print "Matrices are equivalent:", np.array_equal(a, plu_mat)


def plu_full(a_orig):
    a = np.copy(a_orig)
    rows, cols = a.shape
    p = np.eye(rows, cols)
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
        print a
        print "_________________________________t"
        swap_rows(a, k, row_with_max_elem)
        swap_cols(a, k, col_with_max_elem)

        swap_rows(p, k, row_with_max_elem)
        swap_cols(p, k, col_with_max_elem)

        for i in range(k + 1, rows):
            a[i, k] /= a[k, k]
            for j in range(k + 1, cols):
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

    return p, lower, upper


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

    P, L, U = plu_full(B)

    print "Matrix P:"
    print P
    print "Matrix L:"
    print L
    print "Matrix U:"
    print U

    check_plu(B, P, L, U)
    print "Det:", matrix_determinant(P, L, U)
except Exception as e:
    print e
