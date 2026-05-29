import numpy as np


def reg_pen_L2(W, lambdaa):
    return lambdaa * np.sum(np.square(W))


if __name__ == "__main__":
    matrix = np.array([
        [1, 2, 3],
        [3, 2, 1]
    ])

    print(reg_pen_L2(matrix, 1))

