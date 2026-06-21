from engine.tensor import Tensor
import numpy as np


def mse_loss(pred, target):
    assert isinstance(pred, Tensor)
    target = target if isinstance(target, Tensor) else Tensor(target)

    diff = pred - target
    diff_sq = diff ** 2
    total = diff_sq.sum()
    mean = total * (1 / np.size(target.data))

    return mean


def softmax_cross_entropy(logits, target):
    target = target if isinstance(target, Tensor) else Tensor(target)

    # forward pass (numerically stable softmax)
    shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    batch_size = logits.data.shape[0]
    # cross-entropy: -sum(target * log(probs)), averaged over batch
    loss_value = -np.sum(target.data * np.log(probs)) / batch_size

    result = Tensor(data=loss_value, _children=(logits,))

    # backward pass (the simplified combined gradient)
    def _backward():
        logits.grad += (probs - target.data) * result.grad / batch_size
    result._backward = _backward

    return result
