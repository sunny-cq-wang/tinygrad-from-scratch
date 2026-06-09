def update_W(grad, W, step_size=1e-4):
    """
    param grad: K x D
    param W: K x D
    param step_size: learning rate
    """
    return W - step_size * grad