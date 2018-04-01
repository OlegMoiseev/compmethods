import numpy as np
import SLE
import PLU


def inverse(a):
    rows, cols = a.shape

    inv = np.zeros((rows, cols))
    for i in range(rows):
        z = np.zeros((rows, 1))
        z[i, 0] = 1
        s = SLE.solve_lin_eq(a, z)

        for j in range(rows):
            inv[j, i] = s[j]
    return inv


def check(a, inv):
    eye = PLU.matrix_multiplication(a, inv)
    rows, cols = a.shape
    return np.allclose(eye, np.eye(rows, cols), atol=10e-9)
