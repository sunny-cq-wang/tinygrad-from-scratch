# base class for anything that can be a layer or a model
# forward pass method: given input, define a computation and return output
# track parameters: automatically collect all learnable weights inside
# lets the optimizer know what to update

import numpy as np
from engine.tensor import Tensor


class Module:
    def __call__(self, *args, **kwargs): return self.forward(*args)
    def forward(self, *args): raise NotImplementedError
    def parameters(self):
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
        return params


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(out_features, in_features) * (1 / np.sqrt(in_features)))
        self.b = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.W.T + self.b