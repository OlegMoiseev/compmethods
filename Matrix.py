import PLUQ
import PLU
import SLE
import Inverse
import ConditionalNumber

import numpy as np
import random as rand


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Generate random dimensional, matrix & vector:
dim = rand.randint(2, 10)
mat = 10 * np.random.rand(dim, dim) - 5
vec = 10 * np.random.rand(dim, 1) - 5

# If need matrix with string of zeros
for i in range(dim):
    mat[0, i] = 0.
vec[0, 0] = 0

# Check of the PLUQ method:
P, L, U, Q, rank = PLUQ.decompose(mat)
PLUQ.check(mat, P, L, U, Q)

print "Dimensional: ", dim
print "Det: ", PLUQ.matrix_determinant(U)
print "Rank: ", rank

x = SLE.solve_lin_eq(mat, vec)
print mat
print x
print vec
print "Linear Equations, result of checking - zero vector: ", SLE.check(mat, x, vec)

if rank == dim:
    inv = Inverse.inverse(mat)
    print "Inverse matrix: ", Inverse.check(mat, inv)
    print "Conditional number: ", ConditionalNumber.conditional_number(mat)
