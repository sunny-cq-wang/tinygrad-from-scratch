import numpy as np


def unbroadcast(grad, original_shape):
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    for i,dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(data, dtype=np.float64)

        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(data=self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += unbroadcast(result.grad, self.data.shape)
            other.grad += unbroadcast(result.grad, other.data.shape)
        result._backward = _backward

        return result

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(data=self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += unbroadcast(other.data * result.grad, self.data.shape)
            other.grad += unbroadcast(self.data * result.grad, other.data.shape)
        result._backward = _backward

        return result

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(data=self.data @ other.data, _children=(self, other))

        def _backward():
            self.grad += unbroadcast(result.grad @ other.data.swapaxes(-1, -2), self.data.shape)
            other.grad += unbroadcast(self.data.swapaxes(-1, -2) @ result.grad, other.data.shape)
        result._backward = _backward

        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        result = Tensor(data=self.data ** other, _children=(self,))

        def _backward():
            self.grad += result.grad * (other * self.data ** (other - 1))
        result._backward = _backward

        return result

    def exp(self):
        result = Tensor(data=np.exp(self.data), _children=(self,))

        def _backward():
            self.grad += result.grad * np.exp(self.data)
        result._backward = _backward

        return result

    def log(self):
        result = Tensor(data=np.log(self.data), _children=(self,))

        def _backward():
            self.grad += result.grad * (1 / self.data)
        result._backward = _backward

        return result

    def relu(self):
        result = Tensor(data=np.maximum(0, self.data), _children=(self,))

        def _backward():
            self.grad += result.grad * (result.data > 0)
        result._backward = _backward

        return result

    @property
    def T(self):
        result = Tensor(data=self.data.T, _children=(self,))

        def _backward():
            self.grad += result.grad.T
        result._backward = _backward

        return result

    def reshape(self, *shape):
        result = Tensor(data=self.data.reshape(*shape), _children=(self,))

        def _backward():
            self.grad += result.grad.reshape(self.data.shape)

        result._backward = _backward

        return result

    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        result = Tensor(data=self.data.transpose(*axes), _children=(self,))

        def _backward():
            inv_axes = np.argsort(axes)
            self.grad += result.grad.transpose(*inv_axes)

        result._backward = _backward

        return result

    def sum(self, axis=None, keepdims=False):
        result = Tensor(data=self.data.sum(axis=axis, keepdims=keepdims), _children=(self,))

        def _backward():
            grad = result.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += grad * np.ones_like(self.data)

        result._backward = _backward

        return result

    def max(self, axis=None, keepdims=False):
        data = self.data.max(axis=axis, keepdims=True)
        mask = (self.data == data).astype(self.data.dtype)
        mask /= mask.sum(axis=axis, keepdims=True)  # split grad if there's a tie

        out_data = data if keepdims else np.squeeze(data, axis=axis)
        result = Tensor(data=out_data, _children=(self,))

        def _backward():
            grad = result.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += grad * mask

        result._backward = _backward

        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Tensor(other) * (self ** -1)

    def backward(self):

        topo_order = []
        visited = set()

        def build_topo(T):
            if T not in visited:
                visited.add(T)
                for child in T._prev:
                    build_topo(child)
                topo_order.append(T)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for T in reversed(topo_order):
            T._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Tensor(\n\tdata={self.data}, \n\tgrad={self.grad}\n)"