"""micro.py: a tiny autograd engine and neural net, from scratch, in plain Python.

AI Engineering from Scratch, Ep 1: How a Machine Learns.
Adapted from Andrej Karpathy's micrograd (github.com/karpathy/micrograd).
micrograd: The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy. Full notice in LICENSE-micrograd.
"""
import random


class Value:
    """A number that remembers how it was made, so it can pass blame back."""

    def __init__(self, data, parents=(), op=""):
        self.data = data          # the number itself
        self.grad = 0.0           # blame: how much the loss moves if this number moves
        self._parents = parents   # the Values this one was made from
        self._op = op             # how it was made: "+", "*", "**2", "relu"
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():          # a sum passes the same blame to both sides
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():          # a product scales the blame by the OTHER input
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):         # k is a plain number, like 2
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):               # negative in, zero out: the bend
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # put every Value in order (parents before children), then walk it backwards
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0           # the loss blames itself fully
        for v in reversed(order):
            v._backward()

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Neuron:
    """weights times inputs, plus a bias. Optionally bent by a ReLU."""

    def __init__(self, n_inputs, relu=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]   # born random
        self.b = Value(0.0)
        self.use_relu = relu

    def __call__(self, x):
        total = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return total.relu() if self.use_relu else total

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """a row of neurons that all see the same inputs."""

    def __init__(self, n_inputs, n_outputs, relu=True):
        self.neurons = [Neuron(n_inputs, relu) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """layers stacked: every layer bends (ReLU) except the last one."""

    def __init__(self, n_inputs, sizes):
        dims = [n_inputs] + sizes
        self.layers = [Layer(dims[i], dims[i + 1], relu=i < len(sizes) - 1) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = x if isinstance(x, list) else [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
