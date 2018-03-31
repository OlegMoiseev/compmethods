import numpy as np
import Inverse


def norm(a):
    rows, cols = a.shape
    sums = np.zeros((rows, 1))

    for i in range(rows):
        sums[i, 0] = np.sum(a[i])

    return np.max(sums)


def conditional_number(a):
    inv = Inverse.inverse(a)
    return norm(a)*norm(inv)
