import PLUQ
import numpy as np
import random as rand

for i in range(10):
    print "=================================="
    n = rand.randint(1, 10)
    print "Dim: ", n
    A = 10 * np.random.rand(n, n) - 5

    P, L, U, Q, rank = PLUQ.decompose(A)
    print "Rank: ", rank
    PLUQ.check(A, P, L, U, Q)
    print PLUQ.matrix_determinant(U)