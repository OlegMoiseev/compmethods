import QR

import numpy as np
import random as rand


generate_degenerate_matrix = False
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Generate random dimensional, matrix & vector:
dim = rand.randint(2, 10)
mat = 10 * np.random.rand(dim, dim) - 5
vec = 10 * np.random.rand(dim, 1) - 5

# If need matrix with string of zeros
if generate_degenerate_matrix:
    for i in range(dim):
        mat[i, 0] = 0.

Q, R = QR.decompose(mat)
print QR.check(mat, Q, R)

if not generate_degenerate_matrix:
    x = QR.solve_lin_eq(mat, vec)

    print "A: ", mat
    print "X: ", x.T
    print "b: ", vec.T

    print QR.matrix_multiplication(mat, x) - vec
