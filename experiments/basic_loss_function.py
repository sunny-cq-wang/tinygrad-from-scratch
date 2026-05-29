
def loss_per_case(score, y, delta):
    """
    score: f(x, W, b)
    y:      the slot at which the score of the correct label in f(x, W, b) is located
    delta:  constant difference
    """
    res = 0
    for i in range(len(score)):
        if i == y:
            continue
        res += max(0, score[i] - score[y] + delta)
    return res


def loss(scores, Y, delta):
    """
    scores: f(X, W, b)
    Y:      the correct label for each case
    delta:  constant difference
    """
    L = []
    for i in range(len(scores)):
        L.append(loss_per_case(scores[i], Y[i], delta))
    return L


if __name__ == "__main__":
    scores = [3.2, 5.1, -1.7]  # 3 class scores for one image
    y = 0  # correct class is index 0
    delta = 1
    print(loss_per_case(scores, y, delta))