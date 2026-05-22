def linear_nested_mtx_mult(W, x):
    if len(W[0]) != len(x):
        raise ValueError(f"W and x cannot be multiplied, W row: {len(W[0])}, x column: {len(x)}")

    output = [0 for _ in range(len(W))]
    for i in range(len(output)):
        for j in range(len(x)):
            output[i] += W[i][j] * x[j]
    return output

