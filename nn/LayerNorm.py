from engine.tensor import Tensor
import numpy as np
from nn.Module import Module


class LayerNorm(Module):
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(d_model))
        self.beta = Tensor(np.zeros(d_model))

    def __call__(self, x):
        n = x.data.shape[-1]
        mean = x.sum(axis=-1, keepdims=True) / n
        var = ((x - mean) ** 2).sum(axis=-1, keepdims=True) / n
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        out = self.gamma * x_norm + self.beta
        return out

    def parameters(self):
        return [self.gamma, self.beta]


if __name__ == "__main__":
    x = Tensor(np.random.randn(2, 4))
    ln = LayerNorm(4)
    out = ln(x)
    print(out.data)

    out.sum().backward()
    print(ln.gamma.grad)
    print(ln.beta.grad)


