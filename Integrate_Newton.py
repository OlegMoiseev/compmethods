import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import SLE
# 1.18515

a = 1.8
b = 2.3
alpha = 0.
beta = 0.6


def func(x):
    return 3.7 * np.cos(1.5 * x) * np.exp(- 4 * x / 3) + 2.4 * np.sin(4.5 * x) * np.exp(2 * x / 3) + 4


def weight_func(x):
    return 1 / ((x - a)**alpha * (b - x)**beta)


def full_func(x):
    return func(x) * weight_func(x)


print "SciPy: ", integrate.quad(full_func, 1.8, 2.3)
print "WOLFRAM ALPHA: 1.18515\n"

'''fig = plt.figure()

x_arr = np.arange(1.8, 2.3, 0.01)
num_partitions = x_arr.shape
y_arr = np.empty(num_partitions[0])

for i in range(num_partitions[0]):
    y_arr[i] = func(x_arr[i])*weight_func(x_arr[i])

graph1 = plt.plot(x_arr, y_arr)
grid1 = plt.grid(True)
plt.show()'''

meth = 984.121/6 * 0.0059963006023113501973584582773015
print "Methodical error: ", meth


def iqf(start, finish):
    x1 = start
    x2 = (start + finish) / 2
    x3 = finish
    # p(x) = 1 / ((x - a)^alpha * (b - x)^beta))

    nu0 = ((2.3 - x1)**0.4 - (2.3 - x3)**0.4)/0.4
    nu1 = ((2.3 - x3)**1.4 - (2.3 - x1)**1.4)/1.4 + 2.3 * nu0
    nu2 = ((2.3 - x1)**2.4 - (2.3 - x3)**2.4)/2.4 + 2 * 2.3 * nu1 - 2.3 * 2.3 * nu0

    a1 = (nu2 - nu1 * (x2 + x3) + nu0 * x2 * x3) / ((x2 - x1) * (x3 - x1))
    a2 = -(nu2 - nu1 * (x1 + x3) + nu0 * x1 * x3) / ((x2 - x1) * (x3 - x2))
    a3 = (nu2 - nu1 * (x2 + x1) + nu0 * x2 * x1) / ((x3 - x2) * (x3 - x1))

    integral = a1 * func(x1) + a2 * func(x2) + a3 * func(x3)

    return integral


def cqf(num_partitions):
    step_len = (b - a)/num_partitions
    sum = 0
    for i in range(num_partitions):
        sum += iqf(a+step_len*i, a+step_len*(i+1))
    return sum


def cqf_half():
    eps = 1e-6
    multiplier = 2
    num_partitions = 1
    sum1 = cqf(num_partitions)
    sum2 = cqf(num_partitions * multiplier)
    sum3 = cqf(num_partitions * multiplier * multiplier)
    
    m = -np.log((sum3 - sum2) / (sum2 - sum1)) / np.log(multiplier)
    richardson = (sum3 - sum2) / (multiplier**m - 1)
    num_partitions *= multiplier * multiplier
    while abs(richardson) > eps:
        sum1 = sum2
        sum2 = sum3
        num_partitions *= multiplier
        sum3 = cqf(num_partitions)
    
        m = - np.log((sum3 - sum2) / (sum2 - sum1)) / np.log(multiplier)  # convergence rate
        richardson = (sum3 - sum2) / (multiplier**m - 1)
    return sum3 + richardson


def cqf_opt():
    eps = 1e-6
    multiplier = 2
    sum1 = cqf(1)
    sum2 = cqf(2)
    sum3 = cqf(4)
    m = 3
    richardson = (sum3 - sum2) / (multiplier**m - 1.)

    while abs(richardson) > eps:
        h_opt = ((b - a) / multiplier * (eps * (1. - multiplier**(-m))) / abs(sum2 - sum1))**(1. / m)
        num_partitions = np.ceil((b - a) / h_opt)
        sum1 = sum2
        sum2 = sum3
        sum3 = cqf(int(num_partitions))

        richardson = (sum3 - sum2) / (multiplier**m - 1)

    return sum3 + richardson


print "IQF: ", iqf(a, b)
print "CQF half: ", cqf_half()
print "CQF optimal: ", cqf_opt()

# a = 1.8
# b = 2.3
# alpha = 0
# beta = 3 / 5
# integrate: (f(x)/((x - a)^alpha * (b - x)^beta)) dx from a to b
