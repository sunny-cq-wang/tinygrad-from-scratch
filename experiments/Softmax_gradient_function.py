import numpy as np


def compute_gradient_per_case(x, score, y):
    """
    param x: 1 x D
    param score: 1 x K
    param y: correct class
    """
    f = score - np.max(score)
    exp_f = np.exp(f)
    probability = exp_f / np.sum(exp_f)
    probability[y] -= 1
    return np.outer(probability, x)


def compute_gradient(W, X, Y, lambd=0):
    """
    param W: K x D
    param X: N x D
    param Y: 1 x N
    """
    scores = np.matmul(X, W.T) # N x K

    gradient = np.zeros_like(W)
    for i in range(scores.shape[0]):
        gradient += compute_gradient_per_case(X[i], scores[i], Y[i])
    return gradient / X.shape[0] + lambd * W


if __name__ == "__main__":
    np.random.seed(0)
    K, D, N = 3, 4, 5
    W = np.random.randn(K, D)  # 3 x 4
    X = np.random.randn(N, D)  # 5 x 4
    Y = np.array([0, 1, 2, 0, 1])

    grad = compute_gradient(W, X, Y)
    print(grad.shape)  # what do you expect?
    print(grad)