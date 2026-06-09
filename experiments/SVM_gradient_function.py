import numpy as np


def compute_gradient_per_case(W, x, score, y, delta=1):
    """
    param W: K x D
    param x: 1 x D
    param score: 1 x K
    param y: correct class
    """
    L_i = np.zeros(W.shape[0]) # 1 x K
    for i in range(W.shape[0]):
        if i == y: continue
        if score[i] - score[y] + delta > 0:
            L_i[i] = 1
    L_i[y] = -np.sum(L_i)

    return np.outer(L_i, x)


def compute_gradient(W, X, Y, delta=1):
    """
    param W: K x D
    param X: N x D
    param Y: 1 x N
    """
    scores = np.matmul(X, W.T) # N x K

    gradient = np.zeros_like(W)
    for i in range(scores.shape[0]):
        gradient += compute_gradient_per_case(W, X[i], scores[i], Y[i], delta)
    return gradient / X.shape[0]


if __name__ == "__main__":
    np.random.seed(0)
    K, D, N = 3, 4, 5
    W = np.random.randn(K, D)  # 3 x 4
    X = np.random.randn(N, D)  # 5 x 4
    Y = np.array([0, 1, 2, 0, 1])

    grad = compute_gradient(W, X, Y)
    print(grad.shape)  # what do you expect?
    print(grad)