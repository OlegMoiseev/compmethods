import numpy as np


def matrix_multiplication(a, b):
    rows, _ = a.shape
    _, cols = b.shape
    mat = np.zeros((rows, cols))

    for start in range(rows):
        for j in range(cols):
            for k in range(rows):
                mat[start, j] += a[start, k] * b[k, j]
    return mat


def decompose(a_orig):
    r = np.copy(a_orig)
    rows, cols = r.shape
    q = np.eye(rows, cols)
    mid_mat = np.eye(rows, cols)

    for start in range(rows):
        for j in range(1, rows - start):
            if (abs(r[start, start]) > 10e-9) or (abs(r[start+j, start]) > 10e-9):
                norm = (r[start, start]**2 + r[start+j, start]**2)**0.5
                normalized_first = r[start, start] / norm
                normalized_second = r[start+j, start] / norm

                mid_mat[start, start] = normalized_first
                mid_mat[start, start+j] = normalized_second
                mid_mat[start+j, start] = -normalized_second
                mid_mat[start+j, start+j] = normalized_first

                q = matrix_multiplication(mid_mat, q)
                r = matrix_multiplication(mid_mat, r)

                mid_mat[start, start] = 1
                mid_mat[start, start + j] = 0
                mid_mat[start + j, start] = 0
                mid_mat[start + j, start + j] = 1
        mid_mat[start, start] = 1

    return q, r


def solve_lin_eq(a, b):
    q, r = decompose(a)
    x = matrix_multiplication(q, b)
    rows, cols = r.shape

    for i in range(rows-1, 0, -1):
        if abs(r[i, i]) > 10e-9:
            x[i, 0] /= r[i, i]
            for j in range(i-1, -1, -1):
                x[j, 0] -= x[i, 0] * r[j, i]
        else:
            print "SINGULAR!"
            return np.zeros((rows, cols))
    if r[0, 0] > 10e-9:
        x[0, 0] /= r[0, 0]
    else:
        print "SINGULAR!"
        return np.zeros((rows, cols))
    return x


def check(a, q, r):
    print "Q: ", q
    print "R: ", r
    a_check = matrix_multiplication(q.T, r)
    return np.allclose(a, a_check, atol=10e-9)
