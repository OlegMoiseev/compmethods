import numpy as np
import random as rand


def matrix_multiplication(a, b):
    rows, _ = a.shape
    _, cols = b.shape
    matrix = np.zeros((rows, cols))
    try:
        for i in range(rows):
            for j in range(cols):
                for k in range(rows):
                    matrix[i, j] += a[i, k] * b[k, j]
        return matrix
    except:
        print "MULTIPLICATION FAILED!"
        return np.zeros((rows, cols))



def generate_diagonal_dominant():
    # Generate random dimensional, matrix & vector:
    dim = rand.randint(2, 10)
    matrix = 200 * np.random.rand(dim, dim) - 100
    vector = 200 * np.random.rand(dim, 1) - 100

    for i in range(dim):
        max_in_row = 0
        num_max = 0
        for j in range(dim):
            if abs(matrix[i, j]) > max_in_row:
                max_in_row = abs(matrix[i, j])
                num_max = j
        matrix[i, i], matrix[i, num_max] = matrix[i, num_max], matrix[i, i]
    for i in range(dim):
        for j in range(dim):
            if not i == j:
                matrix[i, j] /= dim-1
    return matrix, vector


def generate_positive_definite():
    dim = rand.randint(2, 5)
    matrix = 200 * np.random.rand(dim, dim) - 100
    vector = 200 * np.random.rand(dim, 1) - 100
    return matrix_multiplication(matrix, matrix.T)/1000, vector


def norm_oo(a):
    norm = 0.
    rows, cols = a.shape
    for i in range(rows):
        tmp = abs(a[i]).sum()
        if tmp > norm:
            norm = tmp
    return norm


def jacobi(a, b):
    currency = 10e-12
    max_iterations = 10e3
    rows, cols = a.shape
    h = np.zeros((rows, cols))
    g = np.zeros((rows, 1))
    for i in range(rows):
        for j in range(i):
            h[i, j] = - a[i, j] / a[i, i]
        h[i, i] = 0
        for j in range(i+1, cols):
            h[i, j] = - a[i, j] / a[i, i]
        g[i, 0] = b[i, 0] / a[i, i]
    x_n_1 = np.copy(g)
    counter = 0
    k = norm_oo(h) / (1 - norm_oo(h))
    if k < 0:
        k = 10
    while True:
        x_n = matrix_multiplication(h, x_n_1) + g
        x_n, x_n_1 = x_n_1, x_n
        counter += 1
        if not((k * abs(norm_oo(x_n_1 - x_n))) > currency and counter < max_iterations):
            break
    print "Jacobi iterations: ", counter
    if counter > max_iterations or np.isnan(x_n_1[0, 0]):
        raise Exception("DOESN'T COVERAGE!!!")
    return x_n_1


def seidel(a, b):
    currency = 10e-12
    max_iterations = 10e3
    rows, cols = a.shape
    x_n_1 = np.ones((rows, 1))
    x_n = np.zeros((rows, 1))

    counter = 0

    r = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(i, rows):
            r[i, j] = -a[i, j] / a[i, i]

    h = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            h[i, j] = -a[i, j] / a[i, i]

    k = abs(norm_oo(r) / (1 - norm_oo(h)))

    while True:
        for i in range(rows):
            x_n[i, 0] = b[i, 0] / a[i, i]
            for j in range(i):
                x_n[i, 0] -= x_n[j, 0] * a[i, j] / a[i, i]
            for j in range(i+1, rows):
                x_n[i, 0] -= x_n_1[j, 0] * a[i, j] / a[i, i]

        x_n, x_n_1 = x_n_1, x_n
        counter += 1
        if (k * abs(norm_oo(x_n_1 - x_n))) < currency:
            break

    print "Seidel iterations: ", counter
    return x_n_1


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
np.seterr(all='warn')

mat_pd, vec_pd = generate_positive_definite()
print mat_pd
print vec_pd

mat_dd, vec_dd = generate_diagonal_dominant()
try:
    print "Positive definite - "
    x_j_pd = jacobi(mat_pd, vec_pd)
    print "Diagonal dominant - "
    x_j_dd = jacobi(mat_dd, vec_dd)

except Exception as e:
    print e

else:
    '''print mat
    print vec
    print x'''
    print matrix_multiplication(mat_pd, x_j_pd) - vec_pd
    print matrix_multiplication(mat_dd, x_j_dd) - vec_dd

print "Positive definite - "
x_s_pd = seidel(mat_pd, vec_pd)
print matrix_multiplication(mat_pd, x_s_pd) - vec_pd

print "Diagonal dominant - "
x_s_dd = seidel(mat_dd, vec_dd)
print matrix_multiplication(mat_dd, x_s_dd) - vec_dd
