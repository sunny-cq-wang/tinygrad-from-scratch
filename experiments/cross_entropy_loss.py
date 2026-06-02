import math
import numpy as np


def cross_entropy_per_case(f, y):
    f -= np.max(f)
    exp_f = np.exp(f)
    probability = exp_f / np.sum(exp_f)
    return -np.log(probability[y])


def cross_entropy_loss(F, Y):
    L = []
    for i in range(len(F)):
        L.append(cross_entropy_per_case(F[i], Y[i]))
    return sum(L) / len(L)


if __name__ == "__main__":
    f = np.array([3.2, 5.1, -1.7])
    y = 0
    print(cross_entropy_per_case(f, y))