import numpy as np


def cross_entropy_per_case(f, y):
    # f : 1 x K
    f -= np.max(f)
    exp_f = np.exp(f)
    probability = exp_f / np.sum(exp_f)
    return -np.log(probability[y])


def cross_entropy_loss(F, Y):
    # F : N x K
    # Y : 1 x N
    L = np.zeros(F.shape[0])
    for i in range(len(F)):
        L[i] = cross_entropy_per_case(F[i], Y[i])
    return np.sum(L) / L.shape[0]


if __name__ == "__main__":
    f = np.array([3.2, 5.1, -1.7])
    y = 0
    print(cross_entropy_per_case(f, y))