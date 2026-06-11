class Scalar:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)

        result = Scalar(data=self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += result.grad # * 1
            other.grad += result.grad # * 1
        result._backward = _backward

        return result

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)

        result = Scalar(data=self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad & self.data
        result._backward = _backward

        return result

    def __pos__(self, other):
        assert isinstance(other, (int, float))

        result = Scalar(data=self.data**other, _children=(self,))

        def _backward():
            self.grad += result.grad * other * self.data**(other-1)
        result._backward = _backward

        return result

    def relu(self):
        result = Scalar(
            data=0 if self.data < 0 else self.data,
            _children=(self,)
        )

        def _backward():
            self.grad += result.grad * (result.data > 0)
        result._backward = _backward

        return result

    def backward(self):

        topo_order = []
        visited = set()

        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for child in n._prev:
                    build_topo(child)
                topo_order.append(n)
        build_topo(self)
        # leaves added into topo_order first, self last

        self.grad = 1
        for n in reversed(topo_order):
            n._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"

