# x + lg(x) - 0.5 = 0
# 1 + 1/(x ln 10) = 0

import numpy as np


def func(var):
    return var + np.log10(var) - 0.5


def derivative_func(var):
    return 1 + 1/(var*np.log(10))


accuracy = 10e-4
x = 0.7
count = 0

while True:
    delta = func(x)/derivative_func(x)
    x -= delta
    count += 1
    if abs(delta) < accuracy:
        break

print "X =", x
print "Residual =", func(x)
print "Steps:", count
