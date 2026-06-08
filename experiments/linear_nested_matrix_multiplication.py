import numpy as np
import time

def linear_nested_mtx_mult_vect(W, x):
    """
    param W: dimensions K (labels) and D (pixels)
    param x: dimensions D (pixels) and 1
    return: dimensions K (labels) and 1
    """
    if len(W[0]) != len(x):
        raise ValueError(f"W and x cannot be multiplied, W row: {len(W[0])}, x column: {len(x)}")

    output = [0 for _ in range(len(W))]
    for i in range(len(output)):
        for j in range(len(x)):
            output[i] += W[i][j] * x[j]
    return output


def linear_nested_mtx_mult(W, X):
    """
    param W: dimensions K (labels) and D (pixels)
    param X: dimensions N (images) and D (pixels)
    return: dimensions N (images) and K (labels)
    computes X @ W.T
    """
    if len(W[0]) != len(X[0]):
        raise ValueError(f"W and X cannot be multiplied, W {len(W[0])} != {len(X[0])} X")

    output = []
    for i in range(len(X)):
        output.append(linear_nested_mtx_mult_vect(W, X[i]))
    return output


if __name__ == "__main__":
    W = np.random.randint(0, 10, size=(100, 100))
    X = np.random.randint(0, 10, size=(100, 100))

    a = time.time()
    res1 = linear_nested_mtx_mult(W, X)
    b = time.time()

    c = time.time()
    res2 = np.matmul(X, W.T)
    # X: N x D
    # W: K x D
    # W.t: D x K
    # X x W.T => (N x D) x (D x K) = N x K
    # scores: N x K
    d = time.time()

    print(f"Manual time: {b - a}, Numpy time: {d - c}")