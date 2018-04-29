import numpy as np
import SLE


def get_jacobian(x):
    jac = np.zeros((10, 10))
    jac[0, 0] = -np.sin(x[0] * x[1]) * x[1]
    jac[0, 1] = -np.sin(x[0] * x[1]) * x[0]
    jac[0, 2] = 3.0 * np.exp(-3.0 * x[2])
    jac[0, 3] = x[4] * x[4]
    jac[0, 4] = 2.0 * x[3] * x[4]
    jac[0, 5] = -1.0
    jac[0, 6] = 0.0
    jac[0, 7] = -2.0 * np.cosh(2 * x[7]) * x[8]
    jac[0, 8] = -np.sinh(2.0 * x[7])
    jac[0, 9] = 2.0
    jac[1, 0] = np.cos(x[0] * x[1]) * x[1]
    jac[1, 1] = np.cos(x[0] * x[1]) * x[0]
    jac[1, 2] = x[8] * x[6]
    jac[1, 3] = 0.0
    jac[1, 4] = 6.0 * x[4]
    jac[1, 5] = -np.exp(-x[9] + x[5]) - x[7] - 1.0
    jac[1, 6] = x[2] * x[8]
    jac[1, 7] = -x[5]
    jac[1, 8] = x[2] * x[6]
    jac[1, 9] = np.exp(-x[9] + x[5])
    jac[2, 0] = 1.0
    jac[2, 1] = -1.0
    jac[2, 2] = 1.0
    jac[2, 3] = -1.0
    jac[2, 4] = 1.0
    jac[2, 5] = -1.0
    jac[2, 6] = 1.0
    jac[2, 7] = -1.0
    jac[2, 8] = 1.0
    jac[2, 9] = -1.0
    jac[3, 0] = -x[4] * (x[2] + x[0])**(-2.0)
    jac[3, 1] = -2.0 * np.cos(x[1] * x[1]) * x[1]
    jac[3, 2] = -x[4] * (x[2] + x[0])**(-2.0)
    jac[3, 3] = -2.0 * np.sin(-x[8] + x[3])
    jac[3, 4] = 1.0 / (x[2] + x[0])
    jac[3, 5] = 0.0
    jac[3, 6] = -2.0 * np.cos(x[6] * x[9]) * np.sin(x[6] * x[9]) * x[9]
    jac[3, 7] = -1.0
    jac[3, 8] = 2.0 * np.sin(-x[8] + x[3])
    jac[3, 9] = -2.0 * np.cos(x[6] * x[9]) * np.sin(x[6] * x[9]) * x[6]
    jac[4, 0] = 2.0 * x[7]
    jac[4, 1] = -2.0 * np.sin(x[1])
    jac[4, 2] = 2.0 * x[7]
    jac[4, 3] = (-x[8] + x[3])*(-2.0)
    jac[4, 4] = np.cos(x[4])
    jac[4, 5] = x[6] * np.exp(-x[6] * (-x[9] + x[5]))
    jac[4, 6] = -(x[9] - x[5]) * np.exp(-x[6] * (-x[9] + x[5]))
    jac[4, 7] = 2.0 * x[2] + 2.0 * x[0]
    jac[4, 8] = -(-x[8] + x[3])**(-2.)
    jac[4, 9] = -x[6] * np.exp(-x[6] * (-x[9] + x[5]))
    jac[5, 0] = np.exp(x[0] - x[3] - x[8])
    jac[5, 1] = -1.5 * np.sin(3. * x[9] * x[1]) * x[9]
    jac[5, 2] = -x[5]
    jac[5, 3] = -np.exp(x[0] - x[3] - x[8])
    jac[5, 4] = 2.0 * x[4] / x[7]
    jac[5, 5] = -x[2]
    jac[5, 6] = 0.0
    jac[5, 7] = -x[4] * x[4] * (x[7])**(-2.)
    jac[5, 8] = -np.exp(x[0] - x[3] - x[8])
    jac[5, 9] = -1.5 * np.sin(3. * x[9] * x[1]) * x[1]
    jac[6, 0] = np.cos(x[3])
    jac[6, 1] = 3.0 * x[1] * x[1] * x[6]
    jac[6, 2] = 1.0
    jac[6, 3] = -(x[0] - x[5]) * np.sin(x[3])
    jac[6, 4] = np.cos(x[9] / x[4] + x[7]) * x[9] * (x[4])**(-2.0)
    jac[6, 5] = -np.cos(x[3])
    jac[6, 6] = (x[1])**3.
    jac[6, 7] = -np.cos(x[9] / x[4] + x[7])
    jac[6, 8] = 0.0
    jac[6, 9] = -np.cos(x[9] / x[4] + x[7]) / x[4]
    jac[7, 0] = 2.0 * x[4] * (x[0] - 2. * x[5])
    jac[7, 1] = -x[6] * np.exp(x[1] * x[6] + x[9])
    jac[7, 2] = -2.0 * np.cos(-x[8] + x[2])
    jac[7, 3] = 1.5
    jac[7, 4] = (x[0] - 2. * x[5])**2.0
    jac[7, 5] = -4.0 * x[4] * (x[0] - 2.0 * x[5])
    jac[7, 6] = -x[1] * np.exp((x[1] * x[6]) + x[9])
    jac[7, 7] = 0.0
    jac[7, 8] = 2.0 * np.cos(-x[8] + x[2])
    jac[7, 9] = -np.exp((x[1] * x[6]) + x[9])
    jac[8, 0] = -3.0
    jac[8, 1] = -2.0 * x[7] * x[9] * x[6]
    jac[8, 2] = 0.0
    jac[8, 3] = np.exp((x[4] + x[3]))
    jac[8, 4] = np.exp((x[4] + x[3]))
    jac[8, 5] = -7.0 * (x[5])**(-2.0)
    jac[8, 6] = -2.0 * x[1] * x[7] * x[9]
    jac[8, 7] = -2.0 * x[1] * x[9] * x[6]
    jac[8, 8] = 3.0
    jac[8, 9] = -2.0 * x[1] * x[7] * x[6]
    jac[9, 0] = x[9]
    jac[9, 1] = x[8]
    jac[9, 2] = -x[7]
    jac[9, 3] = np.cos(x[3] + x[4] + x[5]) * x[6]
    jac[9, 4] = np.cos(x[3] + x[4] + x[5]) * x[6]
    jac[9, 5] = np.cos(x[3] + x[4] + x[5]) * x[6]
    jac[9, 6] = np.sin(x[3] + x[4] + x[5])
    jac[9, 7] = -x[2]
    jac[9, 8] = x[1]
    jac[9, 9] = x[0]

    return jac


def get_func(x):
    result = np.zeros((10, 1))
    result[0, 0] = np.cos(x[0] * x[1]) - np.exp(-3.0 * x[2]) + x[3] * x[4] * x[4] - x[5] - np.sinh(2 * x[7])\
                   * x[8] + 2.0 * x[9] + 2.0004339741653854440
    result[1, 0] = np.sin(x[0] * x[1]) + x[2] * x[8] * x[6] - np.exp(-x[9] + x[5]) + 3 * x[4] * x[4] - x[5]\
                   * (x[7] + 1.0) + 10.886272036407019994
    result[2, 0] = x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] - 3.1361904761904761904
    result[3, 0] = 2.0 * np.cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - np.sin(x[1] * x[1]) + np.cos(x[6] * x[9])\
                   * np.cos(x[6] * x[9]) - x[7] - 0.1707472705022304757
    result[4, 0] = np.sin(x[4]) + 2.0 * x[7] * (x[2] + x[0]) - np.exp(-x[6] * (-x[9] + x[5])) + 2.0 * np.cos(x[1])\
                   - 1.0 / (x[3] - x[8]) - 0.3685896273101277862
    result[5, 0] = np.exp(x[0] - x[3] - x[8]) + x[4] * x[4] / x[7] + 0.5 * np.cos(3 * x[9] * x[1]) - x[5] * x[2]\
                   + 2.0491086016771875115
    result[6, 0] = x[1] * x[1] * x[1] * x[6] - np.sin(x[9] / x[4] + x[7]) + (x[0] - x[5]) * np.cos(x[3]) + x[2]\
                   - 0.7380430076202798014
    result[7, 0] = x[4] * (x[0] - 2.0 * x[5]) * (x[0] - 2.0 * x[5]) - 2.0 * np.sin(-x[8] + x[2]) + 1.5 * x[3]\
                   - np.exp(x[1] * x[6] + x[9]) + 3.566832198969380904
    result[8, 0] = 7.0 / x[5] + np.exp(x[4] + x[3]) - 2.0 * x[1] * x[7] * x[9] * x[6] + 3.0 * x[8] - 3.0 * x[0]\
                   - 8.4394734508383257499
    result[9, 0] = x[9] * x[0] + x[8] * x[1] - x[7] * x[2] + np.sin(x[3] + x[4] + x[5]) * x[6] - 0.78238095238095238096

    return -result


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

x = np.array([[0.5], [0.5], [1.5], [-1.0], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])

max_iterations = 90
count = 0
while True:
    F = get_func(x)
    J = get_jacobian(x)

    delta = SLE.solve_lin_eq(J, F)

    max = delta.max()
    x += delta
    count += 1
    if max < 10e-6 or count > max_iterations:
        break

print "Iterations:", count
print x
print get_func(x)
