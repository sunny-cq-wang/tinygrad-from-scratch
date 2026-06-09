import numpy as np


def loss_per_case(score, y, delta=1):
    """
    score: f(x, W, b) --- 1 x K
    y:      the slot at which the score of the correct label in f(x, W, b) is located
    delta:  constant difference
    """
    res = 0
    for i in range(score.shape[0]):
        if i == y:
            continue
        res += max(0, score[i] - score[y] + delta)
    return res


def loss(scores, Y, delta):
    """
    scores: F(X, W, b) --- N x K
    Y:      the correct label for each case --- 1 X N
    delta:  constant difference
    """
    L = np.zeros(scores.shape[0])
    for i in range(scores.shape[0]):
        L[i] = loss_per_case(scores[i], Y[i], delta)
    return np.sum(L) / len(scores)\


if __name__ == "__main__":
    scores = np.array([3.2, 5.1, -1.7])  # 3 class scores for one image
    y = 0  # correct class is index 0
    delta = 1
    print(loss_per_case(scores, y, delta))