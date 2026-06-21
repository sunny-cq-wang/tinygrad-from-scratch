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
        self.data = data
        self.grad = np.zeros_like(data)

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
            self.grad += result.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ result.grad
        result._backward = _backward

        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        result = Tensor(data=self.data ** other, _children=(self,))

        def _backward():
            self.grad += result.grad * (other * self.data ** (other - 1))
        result._backward = _backward

        return result

    def sum(self):
        result = Tensor(data=self.data.sum(), _children=(self,))

        def _backward():
            self.grad += result.grad * np.ones_like(self.data)
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