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
    t = np.eye(rows, cols)

    for start in range(rows):
        for j in range(1, rows - start):
            if (abs(r[start, start]) > 10e-9) or (abs(r[start+j, start]) > 10e-9):
                l = (r[start, start]**2 + r[start+j, start]**2)**0.5
                c = r[start, start] / l
                s = r[start+j, start] / l

                t[start, start] = c
                t[start, start+j] = s
                t[start+j, start] = -s
                t[start+j, start+j] = c

                q = matrix_multiplication(t, q)
                r = matrix_multiplication(t, r)

                t[start, start] = 1
                t[start, start + j] = 0
                t[start + j, start] = 0
                t[start + j, start + j] = 1
        t[start, start] = 1

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
    a_check = matrix_multiplication(q.T, r)
    return np.allclose(a, a_check, atol=10e-9)
